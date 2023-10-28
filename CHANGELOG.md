## Changelog
#### v3.0.0; 2023/10/18
- Added fast inference for tortoise with HiFi Decoder (inspired by xtts by [coquiTTS](https://github.com/coqui-ai/TTS) 🐸, check out their multilingual model for noncommercial uses)
#### v2.8.0; 2023/9/13
- Added custom tokenizer for non-english models
#### v2.7.0; 2023/7/26
- Bug fixes
- Added Apple Silicon Support
- Updated Transformer version
#### v2.6.0; 2023/7/26
- Bug fixes

#### v2.5.0; 2023/7/09
- Added kv_cache support 5x faster
- Added deepspeed support 10x faster
- Added half precision support
  
#### v2.4.0; 2022/5/17
- Removed CVVP model. Found that it does not, in fact, make an appreciable difference in the output.
- Add better debugging support; existing tools now spit out debug files which can be used to reproduce bad runs.

#### v2.3.0; 2022/5/12
- New CLVP-large model for further improved decoding guidance.
- Improvements to read.py and do_tts.py (new options)

#### v2.2.0; 2022/5/5
- Added several new voices from the training set.
- Automated redaction. Wrap the text you want to use to prompt the model but not be spoken in brackets.
- Bug fixes

#### v2.1.0; 2022/5/2
- Added ability to produce totally random voices.
- Added ability to download voice conditioning latent via a script, and then use a user-provided conditioning latent.
- Added ability to use your own pretrained models.
- Refactored directory structures.
- Performance improvements & bug fixes.
