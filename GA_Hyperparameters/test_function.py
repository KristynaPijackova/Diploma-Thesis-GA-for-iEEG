# Test function to simulate the scores and monitor the genetic algorithm

# WINDOW = random.randint(0, 1)
# NPERSEG = random.randint(5, 9)
# NOVERLAP = random.randint(1, 6)
# NFFT = random.randint(0, 5)
# # NN HP
# NFILT = 2 ** random.randint(6, 10)
# NKERN = random.choice([3, 5, 7, 11, 15])
# NHIDDEN = 2 ** random.randint(6, 9)  # GRU hidden
# NGRUS = random.choice([1, 2, 3])  # GRU layers
# # Training HP
# BATCH = random.choice([[64, 128, 256]])
# LR = random.choice([1e-3, 5e-4, 1e-4])
# L2 = random.choice([0, 1e-6, 1e-4])

def test_func(WINDOW, NPERSEG, NOVERLAP, NFFT, NFILT, KERNEL, HIDDEN, NGRUS, BATCH, LR, L2):
    optimal_params = {"WINDOW": 1, "NPERSEG": 6, "NOVERLAP": 1, "NFFT": 2,
                      "NFILT": 64, "KERNEL": 3, "HIDDEN": 128, "NGRUS": 3,
                      "BATCH": 32, "LR": 1e-3, "L2": 1e-4}

    max_params = {"WINDOW": 1, "NPERSEG": 9, "NOVERLAP": 6, "NFFT": 5,
                  "NFILT": 2**10, "KERNEL": 15, "HIDDEN": 2**9, "NGRUS": 3,
                  "BATCH": 256, "LR": 1e-3, "L2": 1e-4}

    params = {"WINDOW": WINDOW, "NPERSEG": NPERSEG, "NOVERLAP": NOVERLAP, "NFFT": NFFT,
              "NFILT": NFILT, "KERNEL": KERNEL, "HIDDEN": HIDDEN, "NGRUS": NGRUS,
              "BATCH": BATCH, "LR": LR, "L2": L2}

    fx_accumulate = 0

    for key in params:
        fx_key = ((params[key] - optimal_params[key]) / max_params[key]) ** 2
        print(fx_key)
        fx_accumulate = fx_accumulate + fx_key

    return -fx_accumulate, 0, 0, 0

def test_func_multi(WINDOW, NPERSEG, NOVERLAP, NFFT, NFILT, KERNEL, HIDDEN, NGRUS, BATCH, LR, L2):
    optimal_params_1 = {"WINDOW": 1, "NPERSEG": 6, "NOVERLAP": 1, "NFFT": 2,
                      "NFILT": 64, "KERNEL": 3, "HIDDEN": 128, "NGRUS": 3,
                      "BATCH": 32, "LR": 1e-3, "L2": 1e-4}

    optimal_params_2 = {"WINDOW": 1, "NPERSEG": 6, "NOVERLAP": 1, "NFFT": 2,
                      "NFILT": 256, "KERNEL": 3, "HIDDEN": 128, "NGRUS": 3,
                      "BATCH": 128, "LR": 5e-4, "L2": 1e-6}

    optimal_params_3 = {"WINDOW": 0, "NPERSEG": 5, "NOVERLAP": 1, "NFFT": 1,
                      "NFILT": 128, "KERNEL": 5, "HIDDEN": 256, "NGRUS": 2,
                      "BATCH": 64, "LR": 1e-4, "L2": 1e-4}

    max_params = {"WINDOW": 1, "NPERSEG": 9, "NOVERLAP": 6, "NFFT": 5,
                  "NFILT": 2**10, "KERNEL": 15, "HIDDEN": 2**9, "NGRUS": 3,
                  "BATCH": 256, "LR": 1e-3, "L2": 1e-4}

    params = {"WINDOW": WINDOW, "NPERSEG": NPERSEG, "NOVERLAP": NOVERLAP, "NFFT": NFFT,
              "NFILT": NFILT, "KERNEL": KERNEL, "HIDDEN": HIDDEN, "NGRUS": NGRUS,
              "BATCH": BATCH, "LR": LR, "L2": L2}

    fx_accumulate = 0
    fx_accumulate_1 = 0
    fx_accumulate_2 = 0
    fx_accumulate_3 = 0
    for key in params:
        fx_key_1 = ((params[key] - optimal_params_1[key]) / max_params[key]) ** 2
        fx_key_2 = ((params[key] - optimal_params_2[key]) / max_params[key]) ** 2
        fx_key_3 = ((params[key] - optimal_params_3[key]) / max_params[key]) ** 2

        fx_accumulate = fx_accumulate + (fx_key_1 + fx_key_2 + fx_key_3)/3
        fx_accumulate_1 = fx_accumulate + fx_key_1
        fx_accumulate_2 = fx_accumulate + fx_key_2
        fx_accumulate_3 = fx_accumulate + fx_key_3
    return -fx_accumulate, fx_accumulate_1, fx_accumulate_2, fx_accumulate_3
