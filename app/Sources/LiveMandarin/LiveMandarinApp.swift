import AppKit
import ServiceManagement
import SwiftUI

@main
struct LiveMandarinApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        MenuBarExtra("LiveMandarin", systemImage: "captions.bubble") {
            MenuContent(model: appDelegate.coordinator.model, coordinator: appDelegate.coordinator)
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let coordinator = Coordinator()

    func applicationDidFinishLaunching(_ notification: Notification) {
        coordinator.showPanel()
    }

    func applicationWillTerminate(_ notification: Notification) {
        coordinator.shutdown()
    }
}

struct MenuContent: View {
    @ObservedObject var model: CaptionModel
    let coordinator: Coordinator

    var body: some View {
        Text(model.statusText).font(.caption)
        Button(model.isRunning ? "Stop captions" : "Start captions") {
            model.isRunning ? coordinator.stop() : coordinator.start()
        }
        .keyboardShortcut("s")
        Button("Show caption panel") { coordinator.showPanel() }
        Button("Clear captions") { model.clear() }.keyboardShortcut("k")
        Divider()
        Picker("Translate", selection: Binding(get: { model.mode }, set: { coordinator.setMode($0) })) {
            ForEach(Mode.allCases) { Text($0.title).tag($0) }
        }
        Toggle("Microphone: my voice & people in the room", isOn: Binding(
            get: { model.includeMic },
            set: { coordinator.setIncludeMic($0) }
        ))
        Toggle("Launch at login", isOn: Binding(
            get: { SMAppService.mainApp.status == .enabled },
            set: { on in try? on ? SMAppService.mainApp.register() : SMAppService.mainApp.unregister() }
        ))
        Divider()
        Button("Quit LiveMandarin") { NSApp.terminate(nil) }.keyboardShortcut("q")
    }
}

/// Appends timestamped lines to ~/Library/Logs/LiveMandarin-app.log for troubleshooting.
enum AppLog {
    private static let url = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Logs/LiveMandarin-app.log")
    private static let queue = DispatchQueue(label: "captions.log")

    static func write(_ text: String) {
        queue.async {
            let line = "\(Date().formatted(date: .omitted, time: .standard)) \(text)\n"
            if let handle = try? FileHandle(forWritingTo: url) {
                handle.seekToEndOfFile()
                handle.write(line.data(using: .utf8)!)
                try? handle.close()
            } else {
                try? line.write(to: url, atomically: true, encoding: .utf8)
            }
        }
    }
}

/// Wires audio capture → WebSocket → caption model, and owns the panel and server lifecycle.
@MainActor
final class Coordinator {
    let model = CaptionModel()
    private let audio = AudioCapture()
    private let mic = MicCapture()
    private let client = CaptionClient()
    private let server = ServerLauncher()
    private var panel: PanelController?
    private var watchdog: Timer?
    private var startTask: Task<Void, Never>?
    private var lastSoundAt = Date()
    private var lastFrameCount = 0
    private var stalledChecks = 0

    init() {
        client.onMessage = { [weak self] message in
            Task { @MainActor in self?.model.handle(message) }
        }
        client.onState = { [weak self] state in
            AppLog.write("server link: \(state)")
            Task { @MainActor in self?.model.connection = state }
        }
        var frames = 0
        audio.onFrame = { [weak self] data, peak in
            guard let self else { return }
            frames += 1
            if frames == 1 || frames % 600 == 0 { AppLog.write("audio: \(frames) frames captured") }
            var peak = peak
            var data = data
            if self.mic.isRunning {  // mix your voice into the same mono stream
                let voice = self.mic.take(data.count / 2)
                data.withUnsafeMutableBytes { raw in
                    let samples = raw.bindMemory(to: Int16.self)
                    for i in 0..<samples.count {
                        samples[i] = Int16(clamping: Int32(samples[i]) + Int32(voice[i]))
                        peak = max(peak, Float(abs(Int32(samples[i]))) / 32767)
                    }
                }
            }
            self.client.send(data)
            let now = Date()
            if peak > 0.01 { self.lastSoundAt = now }
            Task { @MainActor in
                self.model.level = max(Double(peak), self.model.level * 0.7)
                self.model.silentSeconds = now.timeIntervalSince(self.lastSoundAt)
            }
        }
        watchdog = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.checkCapture() }
        }
        client.mode = model.mode.rawValue
        audio.onError = { [weak self] error in
            Task { @MainActor in self?.fail("Audio capture stopped: \(error.localizedDescription)") }
        }
    }

    func showPanel() {
        if panel == nil { panel = PanelController(model: model, coordinator: self) }
        panel?.show()
    }

    func hidePanel() { panel?.hide() }

    func start() {
        guard !model.isRunning else { return }
        AppLog.write("start pressed")
        model.status = .starting("Checking caption server…")
        showPanel()
        startTask = Task {
            do {
                try await server.ensureRunning { [weak self] text in
                    AppLog.write("server: \(text)")
                    Task { @MainActor in self?.model.status = .starting(text) }
                }
                AppLog.write("server ready; starting audio capture")
                model.status = .starting("Requesting system audio access…")
                try Task.checkCancellation()
                try await audio.start()
                AppLog.write("audio capture started")
                if model.includeMic { startMic() }
                lastSoundAt = Date()
                model.status = .running
                client.connect()
            } catch is CancellationError {
                AppLog.write("start cancelled")
            } catch {
                AppLog.write("start failed: \(error)")
                fail(error.localizedDescription)
            }
        }
    }

    func setMode(_ mode: Mode) {
        model.mode = mode
        client.mode = mode.rawValue
        AppLog.write("mode: \(mode.rawValue)")
    }

    func setIncludeMic(_ on: Bool) {
        model.includeMic = on
        guard case .running = model.status else { return }
        on ? startMic() : mic.stop()
    }

    private func startMic() {
        do {
            try mic.start()
            AppLog.write("microphone started")
        } catch {
            AppLog.write("microphone failed: \(error)")
            model.status = .error("Microphone unavailable: \(error.localizedDescription)")
        }
    }

    func stop() {
        AppLog.write("stop pressed")
        startTask?.cancel()
        startTask = nil
        mic.stop()
        model.stopped()          // reflect Stop immediately; the watchdog stands down
        Task {
            await audio.stop()
            client.disconnect()
        }
    }

    /// Capture can stop delivering after sleep or an audio-device change; restart it if so.
    private func checkCapture() {
        guard case .running = model.status else { stalledChecks = 0; return }
        let count = audio.framesDelivered
        stalledChecks = count == lastFrameCount ? stalledChecks + 1 : 0
        lastFrameCount = count
        guard stalledChecks >= 2 else { return }
        stalledChecks = 0
        AppLog.write("audio stalled (\(count) frames); restarting capture")
        Task {
            await audio.stop()
            do { try await audio.start(); AppLog.write("audio capture restarted") }
            catch { fail("Audio capture stopped: \(error.localizedDescription)") }
        }
    }

    func shutdown() {
        client.disconnect()
        server.terminate()
    }

    private func fail(_ message: String) {
        mic.stop()
        Task { await audio.stop() }
        client.disconnect()
        model.status = .error(message)
    }
}
