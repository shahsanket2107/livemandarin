import AppKit
import SwiftUI

/// Floating, always-on-top caption panel. Non-activating, so clicking it never steals focus
/// from the call; visible over full-screen apps and on every Space.
@MainActor
final class PanelController {
    private let panel: NSPanel

    init(model: CaptionModel, coordinator: Coordinator) {
        // Titled (with the title bar hidden) so macOS provides edge/corner resizing.
        panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 460, height: 400),
                        styleMask: [.nonactivatingPanel, .titled, .resizable, .fullSizeContentView],
                        backing: .buffered, defer: false)
        panel.titleVisibility = .hidden
        panel.titlebarAppearsTransparent = true
        for button in [NSWindow.ButtonType.closeButton, .miniaturizeButton, .zoomButton] {
            panel.standardWindowButton(button)?.isHidden = true
        }
        panel.level = .floating
        panel.isFloatingPanel = true
        panel.hidesOnDeactivate = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.isMovableByWindowBackground = true
        panel.minSize = NSSize(width: 380, height: 160)
        panel.contentView = NSHostingView(rootView: PanelView(model: model, coordinator: coordinator))
        panel.setFrameAutosaveName("CaptionPanel")
        if !panel.setFrameUsingName("CaptionPanel"), let screen = NSScreen.main {
            let area = screen.visibleFrame
            panel.setFrameOrigin(NSPoint(x: area.maxX - 440, y: area.maxY - 400))
        }
    }

    func show() { panel.orderFrontRegardless() }
    func hide() { panel.orderOut(nil) }
}

struct PanelView: View {
    @ObservedObject var model: CaptionModel
    let coordinator: Coordinator

    var body: some View {
        VStack(spacing: 0) {
            header
            Rectangle().fill(Color.white.opacity(0.07)).frame(height: 1)
            captions
        }
        .background(GlassBackground())
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color.white.opacity(0.09)))
        .preferredColorScheme(.dark)
    }

    private var header: some View {
        HStack(spacing: 8) {
            StatusDot(model: model).help(model.statusDetail)
            if model.isRunning { LevelMeter(level: model.level).help(model.statusDetail) }
            Text(model.mode.badge)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 7).padding(.vertical, 2)
                .background(Color.white.opacity(0.07), in: Capsule())
                .fixedSize()
            Text(model.statusText)
                .font(.system(size: 11)).foregroundStyle(.secondary)
                .lineLimit(1).truncationMode(.tail).layoutPriority(-1)
            Spacer(minLength: 6)
            HeaderButton("A−", help: "Smaller text") { model.fontSize = max(12, model.fontSize - 1) }
            HeaderButton("A+", help: "Larger text") { model.fontSize = min(32, model.fontSize + 1) }
            HeaderButton("Orig", help: "Show the original sentence under each translation", active: model.showSource) { model.showSource.toggle() }
            HeaderButton(systemImage: "square.and.arrow.down", help: "Save transcript") { saveTranscript() }
            HeaderButton(systemImage: "trash", help: "Clear captions") { model.clear() }
            HeaderButton(systemImage: model.isRunning ? "stop.fill" : "play.fill",
                         help: model.isRunning ? "Stop captions" : "Start captions") {
                model.isRunning ? coordinator.stop() : coordinator.start()
            }
            HeaderButton(systemImage: "xmark", help: "Hide panel (reopen from the menu bar)") { coordinator.hidePanel() }
        }
        .padding(.leading, 14).padding(.trailing, 8).padding(.vertical, 8)
        .frame(height: 40)
    }

    private var captions: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    if case .error(let text) = model.status {
                        Text(text)
                            .font(.system(size: 12)).foregroundStyle(Color(red: 0.99, green: 0.84, blue: 0.39))
                            .frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 10)
                    }
                    if model.entries.isEmpty && !model.isTranscribing {
                        Text(model.isRunning ? "Captions will appear here as people speak."
                                             : "Press ▶ to start captioning whatever is playing on your Mac.")
                            .font(.system(size: 13)).foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity).padding(.vertical, 28)
                    }
                    ForEach(model.entries) { entry in
                        EntryRow(entry: entry, fontSize: model.fontSize, showSource: model.showSource)
                    }
                    if model.isTranscribing {
                        TypingDots().padding(.vertical, 12)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(.horizontal, 14).padding(.vertical, 6)
            }
            .onChange(of: model.entries) { _, _ in proxy.scrollTo("bottom") }
            .onChange(of: model.isTranscribing) { _, _ in proxy.scrollTo("bottom") }
        }
    }

    private func saveTranscript() {
        let dialog = NSSavePanel()
        dialog.nameFieldStringValue = "meet-captions-\(Date().formatted(.iso8601.year().month().day().dateSeparator(.dash))).txt"
        dialog.allowedContentTypes = [.plainText]
        NSApp.activate(ignoringOtherApps: true)
        dialog.begin { response in
            guard response == .OK, let url = dialog.url else { return }
            try? model.transcriptText().write(to: url, atomically: true, encoding: .utf8)
        }
    }
}

private struct EntryRow: View {
    let entry: Entry
    let fontSize: Double
    let showSource: Bool

    private static let colors: [Color] = [
        Color(red: 0.54, green: 0.71, blue: 0.97), Color(red: 0.51, green: 0.85, blue: 0.62),
        Color(red: 0.98, green: 0.74, blue: 0.40), Color(red: 0.93, green: 0.55, blue: 0.75),
        Color(red: 0.66, green: 0.62, blue: 0.95), Color(red: 0.45, green: 0.85, blue: 0.85),
    ]

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            // Speaker identity as a colored stripe: same color, same person.
            RoundedRectangle(cornerRadius: 1.5)
                .fill(entry.speaker.map { Self.colors[$0 % Self.colors.count] } ?? Color.white.opacity(0.12))
                .frame(width: 3)
                .padding(.vertical, 3)
            VStack(alignment: .leading, spacing: 2) {
                if showSource && !entry.src.isEmpty {
                    Text(entry.src).font(.system(size: fontSize * 0.78)).foregroundStyle(.secondary)
                }
                Text(entry.dst).font(.system(size: fontSize)).foregroundStyle(entry.live ? .secondary : .primary)
            }
            .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 6)
        .help(entry.speaker.map { "Speaker \($0 + 1)" } ?? "")
    }
}

private struct StatusDot: View {
    @ObservedObject var model: CaptionModel

    private var color: Color {
        switch model.status {
        case .stopped: return .gray
        case .starting: return .yellow
        case .error: return .red
        case .running: return model.connection == .connected ? .green : .yellow
        }
    }

    var body: some View {
        Circle().fill(color).frame(width: 8, height: 8)
            .overlay(Circle().stroke(color.opacity(0.3), lineWidth: model.speaking ? 4 : 0))
            .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true), value: model.speaking)
    }
}

private struct LevelMeter: View {
    let level: Double

    var body: some View {
        HStack(alignment: .bottom, spacing: 2) {
            ForEach(0..<5, id: \.self) { i in
                let threshold = Double(i + 1) / 6
                RoundedRectangle(cornerRadius: 1)
                    .fill(level >= threshold ? Color.green : Color.white.opacity(0.18))
                    .frame(width: 3, height: 5 + CGFloat(i) * 2)
            }
        }
        .animation(.linear(duration: 0.1), value: level)
    }
}

private struct HeaderButton: View {
    private let label: Text
    private let help: String
    private let active: Bool
    private let action: () -> Void
    @State private var hovering = false

    init(_ title: String, help: String, active: Bool = false, action: @escaping () -> Void) {
        label = Text(title).font(.system(size: 12, weight: .medium))
        self.help = help; self.active = active; self.action = action
    }

    init(systemImage: String, help: String, active: Bool = false, action: @escaping () -> Void) {
        label = Text(Image(systemName: systemImage)).font(.system(size: 11, weight: .semibold))
        self.help = help; self.active = active; self.action = action
    }

    var body: some View {
        Button(action: action) {
            label
                .foregroundStyle(active ? Color(red: 0.54, green: 0.71, blue: 0.97) : (hovering ? .white : .secondary))
                .frame(minWidth: 26, minHeight: 24)
                .background(Color.white.opacity(hovering ? 0.12 : 0), in: RoundedRectangle(cornerRadius: 7))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .help(help)
    }
}

private struct TypingDots: View {
    @State private var on = false

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { i in
                Circle().fill(Color.secondary).frame(width: 6, height: 6)
                    .opacity(on ? 1 : 0.3)
                    .offset(y: on ? -3 : 0)
                    .animation(.easeInOut(duration: 0.45).repeatForever(autoreverses: true).delay(Double(i) * 0.15), value: on)
            }
        }
        .onAppear { on = true }
    }
}

private struct GlassBackground: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = .hudWindow
        view.blendingMode = .behindWindow
        view.state = .active
        view.appearance = NSAppearance(named: .darkAqua)
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {}
}
