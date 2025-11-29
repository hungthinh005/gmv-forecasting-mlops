"""Setup script to initialize project structure"""

import os
from pathlib import Path


def create_directories():
    """Create necessary project directories"""
    directories = [
        "data/raw",
        "data/processed",
        "data/predictions",
        "models",
        "logs",
        "outputs/plots",
        "mlruns",
        "notebooks"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep to preserve empty directories
        gitkeep = path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
        
        print(f"✓ Created {directory}")


def create_env_file():
    """Create .env file from template"""
    env_example = Path("env.example")
    env_file = Path(".env")
    
    if env_file.exists():
        print("✓ .env file already exists")
        return
    
    if env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("✓ Created .env file from template")
    else:
        print("⚠ env.example not found")


def main():
    """Main setup function"""
    print("🚀 Setting up GMV Forecasting MLOps project...")
    print()
    
    # Create directories
    print("Creating project directories...")
    create_directories()
    print()
    
    # Create .env file
    print("Setting up environment configuration...")
    create_env_file()
    print()
    
    print("✅ Project setup complete!")
    print()
    print("Next steps:")
    print("  1. Copy your data to data/raw/data.csv")
    print("  2. Update config/config.yaml with your settings")
    print("  3. Run: python src/data/prepare_data.py")
    print("  4. Run: python src/models/train.py")
    print("  5. Run: python src/evaluation/evaluate.py")
    print("  6. Start API: uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()

