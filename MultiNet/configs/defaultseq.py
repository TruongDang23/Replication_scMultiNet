# Use this function when you use your own data
def makeconfigs(args):
    yourseq = dict()
    encodeLayer1 = list(map(int, args.encodeLayer1))
    encodeLayer2 = list(map(int, args.encodeLayer2))
    decodeLayer1 = list(map(int, args.decodeLayer1))
    decodeLayer2 = list(map(int, args.decodeLayer2))
    discriminateLayer1 = list(map(int, args.discriminateLayer1))
    discriminateLayer2 = list(map(int, args.discriminateLayer2))
    noise_sigma = [args.noise_sigma1, args.noise_sigma2]

    yourseq["encoder_dim"] = [encodeLayer1, encodeLayer2]
    yourseq["decoder_dim"] = [decodeLayer1, decodeLayer2]
    yourseq["hidden_dim"] = args.hidden_dim
    yourseq["discriminator_dim"] = [discriminateLayer1, discriminateLayer2]
    yourseq['activation1'] = args.activation1
    yourseq['activation2'] = args.activation2
    yourseq[f'{args.dataset}_Params'] = {"pretrain_epochs": args.pretrain_epochs,
                                        "train_epochs": args.train_epochs,
                                        "batch_size": args.batch_size,
                                        "cutoff": args.cutoff,
                                        "lr_ae": args.lr_ae,
                                        "lr_d": args.lr_d,
                                        "alpha": args.alpha,
                                        "beta": args.beta,
                                        "sigma": args.sigma,
                                        "gamma": args.gamma,
                                        "zeta_1": args.zeta_1,
                                        "zeta_2": args.zeta_2,
                                        "noise_sigma": noise_sigma,
                                        "phi_1": args.phi_1,
                                        "phi_2": args.phi_2,
                                        "data_preprocess": [args.filter1, args.filter2, args.f1, args.f2]
                                        }
    yourseq['data_preprocess'] = [args.filter1, args.filter2, args.f1, args.f2]
    yourseq["use_indicator"] = False  # default setting
    return yourseq
