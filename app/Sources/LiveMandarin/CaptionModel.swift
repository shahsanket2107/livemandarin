import Foundation

/// One line of the server protocol (see server/app.py).
struct ServerMessage: Decodable {
    let type: String
    let id: Int?
    let src: String?
    let dst: String?
    let dir: String?
    let speaker: Int?
}

struct Entry: Identifiable, Equatable {
    let id: Int
    var src: String
    var dst: String
    var dir: String
    var live: Bool
    var time: String
    var speaker: Int?
}

enum Mode: String, CaseIterable, Identifiable {
    case zhEn = "zh-en", enZh = "en-zh", auto
    var id: String { rawValue }
    var title: String {
        switch self {
        case .zhEn: return "Chinese → English"
        case .enZh: return "English → Chinese"
        case .auto: return "Both directions (auto)"
        }
    }
    var badge: String {
        switch self {
        case .zhEn: return "中 → EN"
        case .enZh: return "EN → 中"
        case .auto: return "中 ⇄ EN"
        }
    }
}

enum Status: Equatable {
    case stopped
    case starting(String)
    case running
    case error(String)
}

@MainActor
final class CaptionModel: ObservableObject {
    @Published var entries: [Entry] = []
    @Published var status: Status = .stopped
    @Published var connection: CaptionClient.State = .disconnected
    @Published var speaking = false
    @Published var pending: Set<Int> = []
    @Published var level: Double = 0          // smoothed audio input level, 0…1
    @Published var silentSeconds: Double = 0  // time since the input last carried sound
    @Published var fontSize: Double {
        didSet { UserDefaults.standard.set(fontSize, forKey: "fontSize") }
    }
    @Published var showSource: Bool {
        didSet { UserDefaults.standard.set(showSource, forKey: "showSource") }
    }
    @Published var includeMic: Bool {
        didSet { UserDefaults.standard.set(includeMic, forKey: "includeMic") }
    }
    @Published var mode: Mode {
        didSet { UserDefaults.standard.set(mode.rawValue, forKey: "mode") }
    }

    init() {
        let defaults = UserDefaults.standard
        fontSize = defaults.object(forKey: "fontSize") as? Double ?? 17
        showSource = defaults.object(forKey: "showSource") as? Bool ?? true
        includeMic = defaults.object(forKey: "includeMic") as? Bool ?? false
        mode = Mode(rawValue: defaults.string(forKey: "mode") ?? "") ?? .zhEn
    }

    var isRunning: Bool {
        if case .stopped = status { return false }
        return true
    }

    var isTranscribing: Bool { speaking || !pending.isEmpty }

    var statusText: String {
        switch status {
        case .stopped: return "Stopped"
        case .starting(let text): return text
        case .error(let text): return text
        case .running:
            switch connection {
            case .connected: return silentSeconds > 5 ? "Live · no audio from your Mac" : "Live"
            case .connecting: return "Connecting to caption server…"
            case .disconnected: return "Caption server offline — retrying"
            }
        }
    }

    /// Server caption ids restart at 1 on every connection, so they are only meaningful within
    /// one session. Map them to app-wide ids; reusing them directly made new captions overwrite
    /// old ones after a reconnect or Stop/Start.
    private var sessionIds: [Int: Int] = [:]
    private var nextLocalId = 0

    private func localId(for serverId: Int) -> Int {
        if let id = sessionIds[serverId] { return id }
        nextLocalId += 1
        sessionIds[serverId] = nextLocalId
        return nextLocalId
    }

    func handle(_ message: ServerMessage) {
        switch message.type {
        case "ready":  // new server session: its ids start over
            sessionIds.removeAll()
        case "listening":
            speaking = true
        case "processing":
            speaking = false
            if let id = message.id { pending.insert(id) }
        case "partial":
            guard let id = message.id else { return }
            pending.remove(id)
            upsert(id: localId(for: id), dst: message.dst ?? "", src: nil, dir: message.dir ?? "", live: true, speaker: message.speaker)
        case "final":
            guard let id = message.id else { return }
            pending.remove(id)
            upsert(id: localId(for: id), dst: message.dst ?? "", src: message.src ?? "", dir: message.dir ?? "", live: false, speaker: message.speaker)
        case "drop":
            if let id = message.id { pending.remove(id) }
        default:
            break
        }
    }

    private func upsert(id: Int, dst: String, src: String?, dir: String, live: Bool, speaker: Int?) {
        if let index = entries.firstIndex(where: { $0.id == id }) {
            entries[index].dst = dst
            if let src { entries[index].src = src }
            entries[index].live = live
            if let speaker { entries[index].speaker = speaker }
        } else {
            let time = Date().formatted(date: .omitted, time: .shortened)
            entries.append(Entry(id: id, src: src ?? "", dst: dst, dir: dir, live: live, time: time, speaker: speaker))
            if entries.count > 300 { entries.removeFirst(entries.count - 300) }
        }
    }

    var statusDetail: String {
        switch status {
        case .running where connection == .connected:
            return "Caption server connected, speech + translation models loaded.\nThe bars show the audio the app hears from your Mac."
        case .running: return "Reconnecting to the caption server…"
        case .starting(let text): return text
        case .error(let text): return text
        case .stopped: return "Press ▶ to start."
        }
    }

    func stopped() {
        status = .stopped
        level = 0
        silentSeconds = 0
        connection = .disconnected
        speaking = false
        pending.removeAll()
    }

    func clear() { entries.removeAll() }

    func transcriptText() -> String {
        entries.filter { !$0.live }
            .map { "[\($0.time)] \($0.speaker.map { "Speaker \($0 + 1): " } ?? "")\($0.dst)\n    \($0.src)" }
            .joined(separator: "\n")
    }
}
