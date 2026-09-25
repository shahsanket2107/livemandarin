import Foundation
import Speech
import Translation

@main struct Probe {
    static func main() async {
        let st = await SpeechTranscriber.supportedLocales.map(\.identifier).sorted()
        print("SpeechTranscriber available:", SpeechTranscriber.isAvailable)
        print("SpeechTranscriber zh/yue locales:", st.filter { $0.hasPrefix("zh") || $0.hasPrefix("yue") || $0.hasPrefix("cmn") })
        print("SpeechTranscriber installed:", await SpeechTranscriber.installedLocales.map(\.identifier))
        let dt = await DictationTranscriber.supportedLocales.map(\.identifier).sorted()
        print("DictationTranscriber zh/yue locales:", dt.filter { $0.hasPrefix("zh") || $0.hasPrefix("yue") })
        let avail = LanguageAvailability()
        for (s, t) in [("zh-Hans", "en"), ("en", "zh-Hans")] {
            let status = await avail.status(from: Locale.Language(identifier: s), to: Locale.Language(identifier: t))
            print("Translation \(s)→\(t):", status)
        }
    }
}
