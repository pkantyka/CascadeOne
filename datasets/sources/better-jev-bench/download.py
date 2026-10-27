"""Download every unmodified raw file from this source; optionally verify offline."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from download_source import main
if __name__=="__main__":
    main("better-jev-bench")
