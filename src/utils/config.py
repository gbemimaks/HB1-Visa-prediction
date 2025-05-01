class FeatureEngineeringConfig:
    def __init__(self):
        self.data_path = '../data/raw/olfc-data.csv'
        self.output_path = '../data/processed/resampled_data.csv'
        self.train_path = '../data/processed/train.csv'
        self.test_path = '../data/processed/test.csv'
        self.label_encoder_pkl = '../src/models/label_encoders.pkl'
        self.scaler_pkl = '../src/models/scaler.pkl'
        self.target_encoder_pkl = '../src/models/target_encoder.pkl'
