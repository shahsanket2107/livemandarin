// Runs Apple's on-device SpeechTranscriber (zh_CN) and Translation (zh-Hans → en) over a
// manifest of labeled clips. Usage: applerun manifest.jsonl out.jsonl
// Each output line: id, set, apple_zh (recognized), asr_s, apple_en_from_ref (translation of
// the reference Chinese, isolating translation quality), apple_en_from_asr, mt_s.
import AVFoundation
import Foundation
import Speech
import Translation

struct Item: Decodable { let set: String; let id: String; let wav: String; let ref_zh: String }

func recognize(_ url: URL, locale: Locale) async throws -> String {
    let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    let collect = Task { () -> String in
        var text = ""
        for try await result in transcriber.results where result.isFinal {
            text += String(result.text.characters)
        }
        return text
    }
    try await analyzer.start(inputAudioFile: AVAudioFile(forReading: url), finishAfterFile: true)
    return try await collect.value
}

/// Translate the `qwen_zh` field of each line with Apple (for the Qwen → Apple pairing).
func translateOnly(input: String, output: String) async throws {
    let session = TranslationSession(installedSource: Locale.Language(identifier: "zh-Hans"),
                                     target: Locale.Language(identifier: "en"), preferredStrategy: .highFidelity)
    try await session.prepareTranslation()
    FileManager.default.createFile(atPath: output, contents: nil)
    let out = try FileHandle(forWritingTo: URL(fileURLWithPath: output))
    for line in try String(contentsOfFile: input, encoding: .utf8).split(separator: "\n") {
        guard let row = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
              row["set"] as? String == "fleurs", let zh = row["qwen_zh"] as? String else { continue }
        let en = zh.isEmpty ? "" : ((try? await session.translate(zh).targetText) ?? "")
        out.write(try JSONSerialization.data(withJSONObject: ["id": row["id"]!, "apple_en": en]) + Data("\n".utf8))
    }
    try out.close()
}

@main struct AppleRun {
    static func main() async throws {
        let args = CommandLine.arguments
        if args[1] == "--translate" { try await translateOnly(input: args[2], output: args[3]); return }
        let items = try String(contentsOfFile: args[1], encoding: .utf8)
            .split(separator: "\n").map { try JSONDecoder().decode(Item.self, from: Data($0.utf8)) }
        let locale = Locale(identifier: "zh_CN")
        let session = TranslationSession(installedSource: Locale.Language(identifier: "zh-Hans"),
                                         target: Locale.Language(identifier: "en"),
                                         preferredStrategy: .highFidelity)
        try await session.prepareTranslation()
        FileManager.default.createFile(atPath: args[2], contents: nil)
        let out = try FileHandle(forWritingTo: URL(fileURLWithPath: args[2]))
        for (n, item) in items.enumerated() {
            var row: [String: Any] = ["id": item.id, "set": item.set]
            do {
                let t0 = Date()
                let zh = try await recognize(URL(fileURLWithPath: item.wav), locale: locale)
                row["asr_s"] = Date().timeIntervalSince(t0)
                row["apple_zh"] = zh
                let t1 = Date()
                row["apple_en_from_ref"] = try await session.translate(item.ref_zh).targetText
                row["mt_s"] = Date().timeIntervalSince(t1)
                row["apple_en_from_asr"] = zh.isEmpty ? "" : try await session.translate(zh).targetText
            } catch {
                row["error"] = "\(error)"
            }
            out.write(try JSONSerialization.data(withJSONObject: row) + Data("\n".utf8))
            if n % 25 == 0 { print("\(n)/\(items.count)") }
        }
        try out.close()
    }
}
