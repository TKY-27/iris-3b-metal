import SwiftUI

/// Generation settings for the next image; defaults match the upstream release.
struct SettingsInspector: View {
    @Environment(Generator.self) private var generator

    var body: some View {
        switch generator.workflow {
        case .create:
            generationSettings
        case .depth:
            WorkflowNote(text: "Iris-3B's depth fine-tune predicts relative depth in one pass. Photos larger than 1024 pixels are downscaled for the model, and the result is resized back. Near is bright; values are affine-invariant, not metric.")
        case .upscale:
            WorkflowNote(text: "Iris-3B's restoration fine-tune enlarges an image 4× and repairs compression and blur in one pass per 1024-pixel tile. Large inputs are first reduced so the output stays within 2048 × 4096, and colors are kept from the original.")
        }
    }

    private var generationSettings: some View {
        @Bindable var generator = generator
        return Form {
            Section("Size") {
                Picker("Size", selection: $generator.size) {
                    Text(ImageSize.square.label).tag(ImageSize.square)
                    Section("Landscape") {
                        ForEach(ImageSize.landscape) { Text($0.label).tag($0) }
                    }
                    Section("Portrait") {
                        ForEach(ImageSize.portrait) { Text($0.label).tag($0) }
                    }
                }
                .labelsHidden()
            }

            Section("Sampling") {
                LabeledContent("Steps") {
                    TextField("Steps", value: $generator.steps, format: .number)
                        .labelsHidden()
                        .multilineTextAlignment(.trailing)
                        .frame(width: 60)
                    Stepper("Steps", value: $generator.steps, in: 1...500, step: 10)
                        .labelsHidden()
                }
                LabeledContent("Guidance") {
                    TextField("Guidance", value: $generator.cfgScale, format: .number.precision(.fractionLength(1)))
                        .labelsHidden()
                        .multilineTextAlignment(.trailing)
                        .frame(width: 60)
                    Stepper("Guidance", value: $generator.cfgScale, in: 1...10, step: 0.5)
                        .labelsHidden()
                }
                Text("100 steps at guidance 3 is the quality the model was evaluated at; fewer steps are faster and change fine detail.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Seed") {
                Toggle("New seed for every image", isOn: $generator.randomizeSeed)
                TextField("Seed", value: $generator.seed, format: .number.grouping(.never))
                    .disabled(generator.randomizeSeed)
            }

            Section {
                DisclosureGroup("Negative prompt") {
                    TextField("Things to avoid", text: $generator.negativePrompt, axis: .vertical)
                        .lineLimit(2...4)
                }
            }
        }
        .formStyle(.grouped)
        .disabled(generator.isBusy)
    }
}

private struct WorkflowNote: View {
    let text: String

    var body: some View {
        Form {
            Section("About") {
                Text(text)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .formStyle(.grouped)
    }
}
