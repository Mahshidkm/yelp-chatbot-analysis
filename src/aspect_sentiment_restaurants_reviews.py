import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from accelerate import Accelerator
import glob
import os

class AspectAnalysisConfig:
    """Configuration class for aspect analysis parameters"""
    def __init__(self):
        self.batch_size = 128
        self.aspect_threshold = 0.86
        self.aspects = [
            'food quality', 'staff behavior', 'cleanliness', 'value for money',
            'convenience', 'wait/delivery time', 'range of choices', 'atmosphere', 'service'
        ]
        self.hypothesis_template = "This review mentions the {} of the restaurant."
        self.output_path = 'my_project/data/processed/yelp_extarct_aspect_and_sentiment_scores_results.csv'

class AspectAnalyzer:
    """Handles aspect detection and sentiment analysis using transformer models"""
    def __init__(self, config):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.aspect_detector, self.absa_model = self._initialize_models()

    def _initialize_models(self):
        """Initialize and configure aspect detection and sentiment analysis models"""
        # Aspect detection pipeline
        aspect_detector = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device_map="auto",
            batch_size=self.config.batch_size
        )

        # Aspect-based sentiment analysis model
        model_name = "yangheng/deberta-v3-base-absa-v1.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model = model.to(self.device)

        absa_model = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            batch_size=self.config.batch_size
        )

        return aspect_detector, absa_model

    def _sentiment_to_score(self, label, confidence):
        """Convert model confidence scores to 1-5 rating scale"""
        if label == "Positive":
            return 5 if confidence >= 0.6 else 4
        elif label == "Negative":
            return 1 if confidence >= 0.7 else 2
        return 3

    def process_batch(self, texts):
        """Process a batch of texts through the full analysis pipeline"""
        batch_results = []
        
        # Aspect detection phase
        with self.accelerator.autocast():
            detection_results = self.aspect_detector(
                texts,
                candidate_labels=self.config.aspects,
                hypothesis_template=self.config.hypothesis_template,
                multi_label=True
            )

        # Prepare inputs for sentiment analysis
        absa_inputs = []
        aspect_records = []
        
        for i, result in enumerate(detection_results):
            relevant_aspects = [
                label for label, score in zip(result['labels'], result['scores'])
                if score > self.config.aspect_threshold
            ]
            aspect_records.append(relevant_aspects)
            
            for aspect in relevant_aspects:
                absa_inputs.append({
                    "text": texts[i],
                    "text_pair": aspect
                })

        # Sentiment analysis phase
        if absa_inputs:
            with self.accelerator.autocast():
                absa_results = self.absa_model(absa_inputs)
            
            ptr = 0
            for aspects_in_review in aspect_records:
                scores = {}
                for aspect in aspects_in_review:
                    result = absa_results[ptr]
                    scores[aspect] = self._sentiment_to_score(result['label'], result['score'])
                    ptr += 1
                batch_results.append(scores)
        else:
            batch_results = [{} for _ in texts]
        
        return batch_results

class ReviewDataset(Dataset):
    """PyTorch Dataset for handling review text data"""
    def __init__(self, texts):
        self.texts = texts
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

class DataProcessor:
    """Handles data loading and preparation using Spark and PyTorch"""
    def __init__(self, config):
        self.config = config

    def load_data(self, data_path):
        """Load and preprocess data from multiple JSON files in a directory."""
        all_files = glob.glob(os.path.join(data_path, "part-*"))
        df_list = [pd.read_json(file, lines=True) for file in all_files]  # Adjust if necessary
        return pd.concat(df_list, ignore_index=True)

    def create_dataloader(self, texts):
        """Create PyTorch DataLoader for the review texts"""
        dataset = ReviewDataset(texts)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() // 2,
            pin_memory=True
        )

class AnalysisPipeline:
    """Orchestrates the complete analysis workflow"""
    def __init__(self, data_path):
        self.config = AspectAnalysisConfig()
        self.data_path = data_path
        self.analyzer = AspectAnalyzer(self.config)
        self.data_processor = DataProcessor(self.config)

    def run(self):
        """Execute the full analysis pipeline"""
        # Load and prepare data
        df = self.data_processor.load_data(self.data_path)
        dataloader = self.data_processor.create_dataloader(df['text'].tolist())
        dataloader = self.analyzer.accelerator.prepare(dataloader)

        # Process batches
        results = []
        for batch in tqdm(dataloader, desc="Processing reviews", 
                        disable=not self.analyzer.accelerator.is_local_main_process):
            texts = [text for text in batch]
            batch_results = self.analyzer.process_batch(texts)
            
            # Collect results from all processes
            all_results = self.analyzer.accelerator.gather_for_metrics(batch_results)
            if self.analyzer.accelerator.is_main_process:
                results.extend(all_results)

        # Save final results
        if self.analyzer.accelerator.is_main_process:
            for aspect in self.config.aspects:
                df[aspect] = [res.get(aspect, None) for res in results]
            df.to_csv(self.config.output_path, index=False)
            print(f"Analysis results saved to {self.config.output_path}")

if __name__ == "__main__":
    pipeline = AnalysisPipeline('my_project/data/cleaned')
    pipeline.run()