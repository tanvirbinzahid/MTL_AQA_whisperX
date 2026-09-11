================================================================================
MTL-AQA — WhisperX Edition (fork of ParitoshParmar/MTL-AQA)
================================================================================

This fork modernizes the original MTL-AQA (CVPR 2019) pipeline and ships a
trained plain-C3D (no self-supervision) best model.

Original authors: Paritosh Parmar (https://github.com/ParitoshParmar)
Fork: tanvirbinzahid/MTL_AQA_whisperX

If you use this code/data, please cite:

  @inproceedings{parmar2019and,
    title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
    author={Parmar, Paritosh and Tran Morris, Brendan},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={304--313},
    year={2019}
  }

WHAT CHANGED vs UPSTREAM
-------------------------------------------------------------------------------
1. WhisperX captions
   - Raw_Annotations/whisper_srt/*.srt : WhisperX transcripts for all 15 videos
   - convert_whisperx_time_aligned_fixed.py : SRT -> final_captions_dict.pkl + vocab.json
   - update_captions.py : merge SRT text + rebuild vocab from train split
   - verify_captions.py : sanity-check the caption dict

2. Modernized training (MTL-AQA_code_release/)
   - train_test_C3DAVG.py : torch.cuda.AMP + GradScaler, gradient clipping,
     MSE logged with final_score_std, dict-format per-epoch stats
     (rho/MSE/5 accuracies), best_model_*.pth naming, weights_only=False
     (torch >= 2.0 safe), optimizer checkpoint save/load
   - opts.py : portable relative paths, C3D-AVG config restored
     (96 frames, 112x112, 171x128 resize, vocab 4768, caption dim 8192)
   - dataloaders/dataloader_C3DAVG.py : fixed sequential frame mapping
   - convert_c3d.py : c3d.pickle -> PyTorch .pth backbone converter

3. Released model (best_nonssl/checkpoints/)
   - Best non-SSL run ("run123", 2026-04-18): Spearman rho 0.8871 on the
     353-dive MTL-AQA test split, classification avg 95.64%
   - Files are Git-LFS tracked (large models)

USAGE
-------------------------------------------------------------------------------
Set options in opts.py, then:
  python train_test_C3DAVG.py            # train
  python3 -u eval_ssl_any.py mtl --ckpt ../best_nonssl/checkpoints   # evaluate

Environment: Python 3.10, PyTorch >= 2.0 (see environment.yml)

Sports-1M pretrained C3D (optional, for training): http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle

Full story and results: see ../README.md