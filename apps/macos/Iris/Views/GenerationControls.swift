import AppKit
import SwiftUI

/// Quality and prompt adherence as named presets on native tick-mark sliders, the exact
/// numbers one `>` away, then the seed and negative prompt.
struct GenerationControls: View {
    @Environment(Workspace.self) private var workspace
    @State private var showsExactSteps = false
    @State private var showsExactGuidance = false

    var body: some View {
        @Bindable var workspace = workspace
        VStack(alignment: .leading, spacing: 16) {
            PresetRow(
                title: "Quality",
                note: "More steps refine detail and take longer. Iris-3B was evaluated at \(GenerationSettings.evaluatedSteps).",
                valueTitle: workspace.settings.qualityTitle,
                presets: QualityPreset.allCases.map(\.title),
                index: QualityPreset.allCases.firstIndex(of: QualityPreset.nearest(to: workspace.settings.steps)) ?? 2,
                select: { workspace.settings.steps = QualityPreset.allCases[$0].steps },
                expanded: $showsExactSteps
            ) {
                ExactNumber(title: "Steps", value: $workspace.settings.steps, range: GenerationSettings.stepRange)
                Text("Fewer steps are faster but soften fine detail such as hands. Use \(GenerationSettings.evaluatedSteps) for final images.")
            }
            Divider()
            PresetRow(
                title: "Prompt Adherence",
                note: "How literally the image follows the prompt (guidance). Very high values can look harsh.",
                valueTitle: workspace.settings.adherenceTitle,
                presets: AdherencePreset.allCases.map(\.title),
                index: AdherencePreset.nearest(to: workspace.settings.cfgScale).rawValue,
                select: { workspace.settings.cfgScale = AdherencePreset.allCases[$0].cfgScale },
                expanded: $showsExactGuidance
            ) {
                ExactGuidance(value: $workspace.settings.cfgScale)
                Text("\(AdherencePreset.balanced.cfgScale.formatted()) is the checkpoint's default.")
            }
            Divider()
            VStack(alignment: .leading, spacing: 8) {
                Toggle("New seed for every image", isOn: $workspace.settings.randomizeSeed)
                if !workspace.settings.randomizeSeed {
                    LabeledContent("Seed") {
                        TextField("Seed", value: $workspace.settings.seed, format: .number.grouping(.never))
                            .labelsHidden()
                            .multilineTextAlignment(.trailing)
                            .frame(width: 110)
                    }
                }
                TextField("Avoid", text: $workspace.settings.negativePrompt, prompt: Text("Things to avoid (negative prompt)"), axis: .vertical)
                    .lineLimit(1...3)
                    .textFieldStyle(.roundedBorder)
            }
            HStack {
                Spacer()
                Button("Restore Defaults") { workspace.settings.restoreDefaults() }
                    .disabled(workspace.settings.hasDefaultTuning)
            }
        }
        .padding(16)
        .frame(width: 320)
    }
}

/// A named preset slider; `>` reveals the exact control.
private struct PresetRow<Exact: View>: View {
    let title: LocalizedStringKey
    let note: String
    let valueTitle: String
    let presets: [String]
    let index: Int
    let select: (Int) -> Void
    @Binding var expanded: Bool
    @ViewBuilder var exact: Exact

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                Text(title).fontWeight(.semibold)
                Spacer()
                Button {
                    expanded.toggle()
                } label: {
                    HStack(spacing: 3) {
                        Text(valueTitle)
                        Image(systemName: "chevron.right")
                            .font(.caption2.weight(.semibold))
                            .rotationEffect(.degrees(expanded ? 90 : 0))
                    }
                    .foregroundStyle(.secondary)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.borderless)
                .help(expanded ? "Hide the exact value" : "Show the exact value")
                .accessibilityLabel("\(valueTitle), exact value")
            }
            TickSlider(count: presets.count, index: index, valueTitles: presets, select: select)
                .accessibilityLabel(Text(title))
            HStack {
                Text(presets.first ?? "")
                Spacer()
                Text(presets.last ?? "")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .accessibilityHidden(true)
            Text(note)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if expanded {
                VStack(alignment: .leading, spacing: 6) { exact }
                    .font(.callout)
                    .foregroundStyle(.secondary)  // the explanation; the fields stay primary
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 4)
            }
        }
    }
}

/// A native slider that rests only on its tick marks. Every user change selects a preset,
/// including the nearest one when a custom number is set.
struct TickSlider: NSViewRepresentable {
    let count: Int
    let index: Int
    let valueTitles: [String]
    let select: (Int) -> Void

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> NSSlider {
        let slider = NSSlider(
            value: Double(index), minValue: 0, maxValue: Double(max(count - 1, 1)),
            target: context.coordinator, action: #selector(Coordinator.changed(_:))
        )
        slider.numberOfTickMarks = count
        slider.allowsTickMarkValuesOnly = true
        slider.isContinuous = true
        return slider
    }

    func updateNSView(_ slider: NSSlider, context: Context) {
        context.coordinator.select = select
        if slider.integerValue != index { slider.integerValue = index }
        slider.setAccessibilityValueDescription(valueTitles.indices.contains(index) ? valueTitles[index] : nil)
    }

    @MainActor
    final class Coordinator: NSObject {
        var select: (Int) -> Void = { _ in }

        @objc func changed(_ slider: NSSlider) {
            select(slider.integerValue)
        }
    }
}

private struct ExactNumber: View {
    let title: LocalizedStringKey
    @Binding var value: Int
    let range: ClosedRange<Int>

    var body: some View {
        LabeledContent(title) {
            HStack(spacing: 4) {
                TextField(title, value: $value, format: .number.grouping(.never))
                    .labelsHidden()
                    .multilineTextAlignment(.trailing)
                    .frame(width: 56)
                Stepper(title, value: $value, in: range)
                    .labelsHidden()
            }
        }
        .foregroundStyle(.primary)
        .onChange(of: value) { _, new in
            if !range.contains(new) { value = new.clamped(to: range) }
        }
    }
}

private struct ExactGuidance: View {
    @Binding var value: Double

    var body: some View {
        LabeledContent("Guidance scale") {
            HStack(spacing: 4) {
                TextField("Guidance scale", value: $value, format: .number.precision(.fractionLength(1...2)))
                    .labelsHidden()
                    .multilineTextAlignment(.trailing)
                    .frame(width: 56)
                Stepper("Guidance scale", value: $value, in: GenerationSettings.guidanceRange, step: 0.25)
                    .labelsHidden()
            }
        }
        .foregroundStyle(.primary)
        .onChange(of: value) { _, new in
            if !GenerationSettings.guidanceRange.contains(new) { value = new.clamped(to: GenerationSettings.guidanceRange) }
        }
    }
}
