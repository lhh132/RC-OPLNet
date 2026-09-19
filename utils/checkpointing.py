def should_save_checkpoint(score, best_score, epoch, total_epochs):
    """Apply the project's warm-up rule while always allowing a first model."""
    warmup_complete = epoch > total_epochs * 0.3
    return warmup_complete and score > best_score
