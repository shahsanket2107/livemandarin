import Foundation

/// Starts the Python caption server (scripts/run.sh) if it isn't already running, and waits
/// until its models are loaded. The project directory is baked into Info.plist at build time.
final class ServerLauncher {
    enum Health { case ready, loading, down }
    enum LauncherError: LocalizedError {
        case noServerDir, timeout
        var errorDescription: String? {
            switch self {
            case .noServerDir: return "The app doesn't know where the caption server is. Rebuild with scripts/build-app.sh."
            case .timeout: return "The caption server didn't become ready. See ~/Library/Logs/LiveMandarin.log."
            }
        }
    }

    private var process: Process?
    private let serverDir = Bundle.main.infoDictionary?["CaptionServerDir"] as? String ?? ""

    func health() async -> Health {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:8765/health")!)
        request.timeoutInterval = 2
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode == 200 ? .ready : .loading
        } catch {
            return .down
        }
    }

    func ensureRunning(progress: @escaping (String) -> Void) async throws {
        if await health() == .down {
            try launch()
            progress("Starting caption server…")
        }
        for _ in 0..<180 {
            switch await health() {
            case .ready: return
            case .loading: progress("Loading speech and translation models…")
            case .down: progress("Starting caption server…")
            }
            try await Task.sleep(for: .seconds(1))
        }
        throw LauncherError.timeout
    }

    private func launch() throws {
        guard !serverDir.isEmpty, serverDir != "__SERVER_DIR__" else { throw LauncherError.noServerDir }
        let logURL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/LiveMandarin.log")
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        let log = try FileHandle(forWritingTo: logURL)
        log.seekToEndOfFile()

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["\(serverDir)/scripts/run.sh"]
        process.standardOutput = log
        process.standardError = log
        try process.run()
        self.process = process
    }

    func terminate() {
        process?.terminate()
        process = nil
    }
}
