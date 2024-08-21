from models.tent import setup_tent

def build_model(model, 
                cfg,
                logger,
                ):
    
    if cfg.adaptation != 'TENT':
        raise NotImplementedError
    tent_model = setup_tent(model, cfg, logger)
    
    return tent_model