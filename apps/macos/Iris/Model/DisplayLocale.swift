import Foundation

extension Locale {
    static var iris: Locale {
        displayLocale(language: Bundle.main.preferredLocalizations.first ?? "en")
    }

    static func displayLocale(language: String, regional: Locale = .current) -> Locale {
        var components = Locale.Components(locale: regional)
        components.languageComponents.languageCode = Locale.LanguageCode(language)
        return Locale(components: components)
    }
}
