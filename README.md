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

## Demo

Example is model that trained from 2016-10-20 news articles with n_units=5000

	sent = '오패산터널 총격전 용의자 검거 서울 연합뉴스 경찰 관계자들이 19일 오후 서울 강북구 오패산 터널 인근에서 사제 총기를 발사해 경찰을 살해한 용의자 성모씨를 검거하고 있다 성씨는 검거 당시 서바이벌 게임에서 쓰는 방탄조끼에 헬멧까지 착용한 상태였다'

	encoder.tokenize(sent)

Result is

	오 패 산 터 널_ 총 격 전_ 용 의자 _ 검 거_ 서울_ 연합뉴스_ 경찰_ 관계 자들이_ 19일_ 오후_ 서울_ 강 북 구_ 오 패 산_ 터 널_ 인근 에서_ 사제 _ 총 기를_ 발사 해_ 경찰 을_ 살 해 한_ 용 의자 _ 성 모 씨를_ 검 거 하고_ 있다_ 성 씨는_ 검 거_ 당시_ 서 바이 벌_ 게임 에서_ 쓰 는_ 방탄 조 끼 에_ 헬 멧 까지_ 착 용한_ 상태 였다_
