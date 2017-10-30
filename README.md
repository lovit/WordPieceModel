# Byte Pair Encoder

For train, 

	from bytepairencoder import BytePairEncoder

	n_units = 5000
	encoder = BytePairEncoder(n_units)
	encoder.train(corpus)

For tokenization, 

	tokens = encoder.tokenize(sent)

For save / load, 

	encoder.save(model_name)

	loaded_encoder = BytePairEncoder()
	loaded_encoder.load(model_fname)

Model saves three parameters, (1) n_units, (2) maximum length of units, (3) dictionary of {unit:frequency}