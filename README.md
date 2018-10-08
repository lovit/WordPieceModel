# Word Piece Model (WPM) light version

For train, 

	from bytepairencoder import BytePairEncoder

	n_iters = 5000
	encoder = BytePairEncoder(n_iters)
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

	오패산터널_ 총격전_ 용의자_ 검거_ 서울_ 연합뉴스_ 경찰_ 관계자들이_ 19일_ 오후_ 서울_ 강북구_ 오패산_ 터널_ 인근에서_ 사제_ 총기를_ 발사해_ 경찰을_ 살해한_ 용의자_ 성 모 씨를_ 검거하고_ 있다_ 성씨는_ 검거_ 당시_ 서바이벌_ 게임에서_ 쓰는_ 방탄조끼에_ 헬멧까지_ 착용한_ 상태였다_ 독자제공_ 영상_ 캡처_ 연합뉴스_ 서울_ 연합뉴스_ 김 은 경_ 기자_ 사제_ 총기로_ 경찰을_ 살해한_ 범 인_ 성 모_ 4 6_ 씨는_ 주도 면밀 했다_ 경찰에_ 따르면_ 성씨는_ 19일_ 오후_ 강북경찰서_ 인근_ 부 ...
