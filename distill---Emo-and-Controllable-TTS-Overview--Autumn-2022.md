# Emo / Controllable TTS Overview

## New Emo Datasets

### Chinese

- Opencpop: A High-Quality Open Source Chinese Popular Song Corpus for Singing Voice Synthesis

### Other languages

- English Dialogues, Conversational Speech: [DailyTalk](https://github.com/keonlee9420/DailyTalk)

## Old Storytelling Datasets

- Blizzard Challenge 2013 -- Catherine Bayers

## Long Form Reading (LFR) / Discourse Synthesis / Storytelling / Context in TTS

- [x] [ICASSP 2022, Towards Expressive Speaking Style Modelling with Hierarchical Context Information for Mandarin Speech Synthesis](https://ieeexplore.ieee.org/document/9747438)

  - authors: China (academic)
  - language = Mandarin
  - dataset: single speaker, online lectures
  - core architecture: FastSpeech2
  - length regulator: after variance preictor --> predict variations at phone level is better than at frame level (speech naturalness improves);
  
  - [demo page](https://thuhcsi.github.io/icassp2022-expressive-tts-hierarchical-context/)

  - significant naturalness and expressivity improvement

  - use context in adjacent sentences
  - better text-representation as input:
  -     - XLNet instead of BERT;
  -     - can process longer texts   
          without length limitation
  - inter-phrase and inter-sentence relationships
  - hierarchical context encoder:
  -     - predict style embeddings   
             from past, present and future sentences
        - RNNs are not enough to capture long-term relations;
        - add (inter-sentence) attention on top of GRU;
  - three-stage training:
  -     - 1) train acoustic model and reference style encoder;
        - 2) train hierarchical context encoder guided by reference-encoder predictions: predict style embeddings from neighbor sentences;
        - 3) train the model jointly;
        
  - 50-60% preference to baseline FastSpeech2;
  - 

- [~] [ICASSP 2022, Discourse-Level Prosody Modeling with a Variational Autoencoder for Non-Autoregressive Expressive Speech Synthesis](https://ieeexplore.ieee.org/document/9746238)

  - authors: China (academic)
  - language = Mandarin

  - dataset: Chinese audiobook reading   
              (single speaker, scraped from the internet)
  - core =  FastSpeech
  - outperformed FastSpeech2
  - discourse-level prosody modeling with a variational autoencoder (VAE)
  - context representation: 
  -    - BERT embeddings
  -    - discourse-level text features   
         (+-K sentences from the context --> BERT)
         
  - strong preference for using more context: slide 18 in the ppt
  

- [ ] [Interspeech 2022, Empathetic Dialogue Speech](https://arxiv.org/pdf/2206.08039.pdf)

   - authors: Japan
   
   - use dialogue history --> context representations --> drive speech style;
   

- [x] [Interspeech 2022, Predicting VQVAE-based Character Acting Style from Quotation-Annotated Text for Audiobook Speech Synthesis](https://www.isca-speech.org/archive/pdfs/interspeech_2022/nakata22_interspeech.pdf)

   - authors: Japan   
             (The University of Tokyo   
             + Nippon Telegraph and Telephone Corporation)
             
   - [demo samples](https://wataru-nakata.github.io/is2022-audiobook/)
   
   - predict character styles to synthesize audiobooks;
   - predict character-appropriate voices;
   
   - demo samples sound as if the attempt at auto-predicting character voices was a **failure** !