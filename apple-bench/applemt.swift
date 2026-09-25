// Apple Translation (zh-Hans → en, highFidelity) over rows from `eval_mt.py --dump`.
// For "presub" rows, the English glossary terms already substituted into the source are marked
// with the skipsTranslation attribute so Apple leaves them untouched.
// Usage: applemt rows.jsonl data/mt_apple.jsonl
import Foundation
import Translation

struct Row: Decodable { let id: String; let strategy: String; let src: String; let protect: [String] }

@main struct AppleMT {
    static func main() async throws {
        let args = CommandLine.arguments
        let rows = try String(contentsOfFile: args[1], encoding: .utf8).split(separator: "\n")
            .map { try JSONDecoder().decode(Row.self, from: Data($0.utf8)) }
        let session = TranslationSession(installedSource: Locale.Language(identifier: "zh-Hans"),
                                         target: Locale.Language(identifier: "en"), preferredStrategy: .highFidelity)
        try await session.prepareTranslation()
        _ = try await session.translate("你好。")
        FileManager.default.createFile(atPath: args[2], contents: nil)
        let out = try FileHandle(forWritingTo: URL(fileURLWithPath: args[2]))
        for row in rows {
            var source = AttributedString(row.src)
            for term in row.protect {
                var searchStart = source.startIndex
                while let range = source[searchStart...].range(of: term) {
                    source[range].translation.skipsTranslation = true
                    searchStart = range.upperBound
                }
            }
            let t0 = Date()
            let english: String
            if row.protect.isEmpty {
                english = (try? await session.translate(row.src).targetText) ?? ""
            } else {
                english = (try? await session.translate(source)).map { String($0.targetText) } ?? ""
            }
            let line: [String: Any] = ["id": row.id, "strategy": row.strategy, "en": english,
                                       "s": Date().timeIntervalSince(t0)]
            out.write(try JSONSerialization.data(withJSONObject: line) + Data("\n".utf8))
        }
        try out.close()
    }
}
