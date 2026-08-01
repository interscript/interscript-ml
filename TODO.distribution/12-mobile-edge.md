# 12 — Mobile + edge variants

**Status:** SPECIFICATION
**Priority:** P7

## Goal

Mobile and edge devs (iOS, Android, React Native, embedded) can drop
a 1-2MB Interscript model into their app and have it run offline in
<30ms/word on a mid-range phone.

## Format matrix

For each task, ship:

| Variant | Size | Latency | Target |
|---|---|---|---|
| fp32 | ~6 MB | 22ms/word | Desktop, server |
| fp16 | ~3 MB | 18ms/word | GPU (browser WebGL2, server) |
| q8 (int8) | ~1.5 MB | 28ms/word | Mobile (default) |
| q4 (int4) | ~0.9 MB | 42ms/word | Constrained devices (Watch, IoT) |
| tflite (int8) | ~1.6 MB | 26ms/word | Android ML (TensorFlow Lite) |
| coreml (mlmodel) | ~1.7 MB | 24ms/word | iOS (Core ML) |

Mobile devs pick the format that matches their runtime.

## Quantization pipeline

`.github/workflows/release.yml` produces all variants from the same
fp32 source:

```yaml
- name: Quantize int8 (ONNX dynamic)
  run: |
    python -m onnxruntime.quantization.quantize \
      --input rababa_arabic.onnx \
      --output rababa_arabic-q8.onnx \
      --quant_format QDQ \
      --per_channel \
      --weight_type QInt8

- name: Quantize int4 (ONNX MatMul 4-bit)
  run: |
    python -m onnxruntime.quantization.quantize \
      --input rababa_arabic.onnx \
      --output rababa_arabic-q4.onnx \
      --quant_format QOperator \
      --weight_type Q4

- name: Convert to TFLite
  run: |
    onnx2tf --input-fp32 rababa_arabic.onnx --output rababa_arabic.tflite
    tflite_quantize --input rababa_arabic.tflite --output rababa_arabic-q8.tflite --mode int8

- name: Convert to Core ML
  run: |
    coremltools.convert(rababa_arabic.onnx).save("rababa_arabic.mlmodel")
```

## Quality impact

Quantization always loses some accuracy. Documented per variant in
`benchmarks.json`:

```
DER (fp32): 4.8%
DER (q8):   5.1%   (+0.3pp)
DER (q4):   6.4%   (+1.6pp)
```

Mobile devs pick the smallest variant that meets their DER budget.

## Runtime SDK per platform

### iOS

```swift
import OnnxRuntimeMobile

let model = try ORTModel(path: Bundle.main.url(forResource: "rababa_arabic-q8", withExtension: "onnx")!)
let output = try model.run(input: "كتب")
// → "كَتَبَ"
```

Or via Core ML:
```swift
let model = try RababaArabic(configuration: MLModelConfiguration())
let output = try model.prediction(input: "كتب")
```

### Android (Kotlin)

```kotlin
val env = OrtEnvironment.getEnvironment()
val session = env.createSession(modelPath)
val result = session.run(Input("كتب".toTensor()))
```

Or via TFLite:
```kotlin
val tflite = Interpreter(loadModelFile("rababa_arabic-q8.tflite"))
val output = doInference(tflite, "كتب")
```

### React Native

```typescript
import { transliterateAsync } from "interscript-ts/mobile"
const result = await transliterateAsync("var-ara-Arab-Arab-rababa", "كتب")
```

`interscript-ts/mobile` uses onnxruntime-react-native under the hood.

## Mobile SDK package

```
@interscript/mobile-rababa-arabic/
├── ios/
│   ├── RababaArabic.framework        (static framework with embedded mlmodel)
│   └── RababaArabic.xcframework      (Xcode 15+)
├── android/
│   ├── rababa_arabic.aar              (Android library)
│   └── rababa_arabic.tflite           (raw model)
├── react-native/
│   └── index.ts
└── package.json
```

## Demo app

`apps/mobile-demo/` directory contains a minimal React Native app:

```typescript
import { transliterateAsync } from "interscript-ts/mobile"
import { useState } from "react"

export function App() {
  const [text, setText] = useState("كتب")
  const [output, setOutput] = useState("")
  return (
    <>
      <TextInput value={text} onChangeText={setText} />
      <Button title="Diacritize" onPress={async () => {
        setOutput((await transliterateAsync("var-ara-Arab-Arab-rababa", text)).output)
      }} />
      <Text>{output}</Text>
    </>
  )
}
```

Publish as TestFlight + Play Store internal track for stakeholders.

## Acceptance

- [ ] int8 + int4 quantization in release workflow
- [ ] TFLite variant produced for Android
- [ ] Core ML variant produced for iOS
- [ ] Mobile SDK package skeleton
- [ ] Demo app builds + runs on iOS + Android
- [ ] p95 < 30ms on mid-range phone (Pixel 6 / iPhone 12)
