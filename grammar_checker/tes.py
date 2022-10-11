from gramformer import Gramformer
import torch
# import spacy.cli 
# spacy.cli.download("en")

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)


gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector

influent_sentences = [
    "Our aim for this research is to build the illumination dome with affordable components for multiple purposes of research and to prove its feasibility. The illumination dome setup is designed to provide a stable lighting environment. For applications like 3D reconstruction through images alone, it is proven to be more challenging to do and provide more inconsistent results if the lighting environment is not controlled. The ambient lighting plays a significant role as it makes the differences from the reflected light of the observed object less intense and, as a result, less detailed. To improve the results, one may subtract the illuminated images and only use the images taken with ambient lighting. but is not the best solution as some features may get underexposed from the subtraction as the object may not have been illuminated uniformly. Therefore, a controlled environment provides better results."
]   

for influent_sentence in influent_sentences:
    corrected_sentences = gf.correct(influent_sentence, max_candidates=1)
    print("[Input] ", influent_sentence)
    for corrected_sentence in corrected_sentences:
      print("[Correction] ",corrected_sentence)
    print("-" *100)