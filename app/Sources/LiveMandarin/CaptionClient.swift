import Foundation

/// WebSocket link to the local caption server: PCM frames out, caption JSON in.
final class CaptionClient: NSObject, URLSessionWebSocketDelegate {
    enum State { case disconnected, connecting, connected }

    var onMessage: ((ServerMessage) -> Void)?
    var onState: ((State) -> Void)?
    /// Translation direction, sent on every (re)connect and whenever it changes.
    var mode: String = "zh-en" { didSet { sendConfig() } }

    private let url = URL(string: "ws://127.0.0.1:8765/ws")!
    private lazy var session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    private var task: URLSessionWebSocketTask?
    private var wanted = false
    private var backoff: TimeInterval = 0.5

    func connect() {
        wanted = true
        backoff = 0.5
        open()
    }

    func disconnect() {
        wanted = false
        task?.send(.string(#"{"type":"flush"}"#)) { _ in }
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
        onState?(.disconnected)
    }

    func send(_ data: Data) {
        task?.send(.data(data)) { _ in }
    }

    private func sendConfig() {
        task?.send(.string(#"{"type":"config","mode":"\#(mode)"}"#)) { _ in }
    }

    private func open() {
        guard wanted else { return }
        onState?(.connecting)
        var request = URLRequest(url: url)
        request.setValue("app://livemandarin", forHTTPHeaderField: "Origin")
        let task = session.webSocketTask(with: request)
        self.task = task
        task.resume()
        receive(on: task)
    }

    private func receive(on task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            guard let self, self.task === task else { return }
            switch result {
            case .success(let message):
                if case .string(let text) = message, let data = text.data(using: .utf8),
                   let decoded = try? JSONDecoder().decode(ServerMessage.self, from: data) {
                    self.onMessage?(decoded)
                }
                self.receive(on: task)
            case .failure:
                self.lost()
            }
        }
    }

    private func lost() {
        task = nil
        guard wanted else { return }
        onState?(.disconnected)
        let delay = backoff
        backoff = min(backoff * 2, 5)
        DispatchQueue.global().asyncAfter(deadline: .now() + delay) { [weak self] in self?.open() }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        backoff = 0.5
        sendConfig()
        onState?(.connected)
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        if task === webSocketTask { lost() }
    }
}
