from gatr.experiments.nbody.experiment import NBodyExperiment
from omegaconf import OmegaConf

def main():
    # Load configuration
    cfg = OmegaConf.load("configs/gmcnn_config.yaml")
    
    # Create experiment
    experiment = NBodyExperiment(cfg)
    
    # Train model
    experiment.train()

if __name__ == "__main__":
    main() 