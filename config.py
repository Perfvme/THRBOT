TELEGRAM_TOKEN = "7714012275:AAEa9TAsBG_A46Y2bj-B-Vqa9YL_jHK9M_4"
BINANCE_API_KEY = "ISxxyCWL1PrgfuCWvuBsVJW07jONxX7HlumEdSYygw41Ka9V3sw3fyKTPBgFD77E"
BINANCE_SECRET_KEY = "qm1Hfis14DKN0Fu2gJqa3LDEZ3gU8doeHxz33Do5Hn4FfJA5W9yA9Y2nGhTjitFx"
GEMINI_API_KEY = "AIzaSyD3BZd4PwO96r8WiCsMRb_sw0HndJYxndk"
ML_CONFIG = {
    'train_symbols': ['BTCUSDT', 'ETHUSDT'],
    'train_interval': 43200,  # 12 hours
    'seq_length': 180,        # 3 hours of 1m data
    'max_features': 20,
    'n_jobs': 1               # 2vCPU optimized
}