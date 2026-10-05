# fly-dcss

초파리 커넥톰을 이용해 **Dungeon Crawl Stone Soup 0.17.1**을 플레이해 보는 프로젝트입니다.

현재 로컬 게임 관측 → CPU 회로 → 이동·대기 → 결과 → 국소 가중치 갱신을 연결한
최소 실험 루프가 동작합니다. 짧은 실행 확인 단계이며, 학습으로 플레이가 좋아진다는
증거는 아직 없습니다.

전체 주석 뉴런(뇌·VNC) 실행과 로컬 인계는 [로컬 안내](docs/LOCAL_HANDOFF.md),
데이터 범위는 [MaleCNS](docs/MALECNS_DATA.md)를 참조하세요.
기존 소규모 실행은 [DCSS 루프](docs/DCSS_LOOP.md), 회로 수식은
[CPU 프로토타입](docs/CIRCUIT_PROTOTYPE.md)에 정리했습니다.
