# 작은 회로 CPU 프로토타입

이 문서는 첫 단계의 회로 단독 검증 기록입니다. 이후 추가된 실제 게임 연동과
커넥톰 유래 소규모 실행은 [DCSS 루프](DCSS_LOOP.md)를 참조하세요.

DCSS 0.17.1 에이전트를 만들기 전, 신경 상태 갱신과 국소 가중치 수정을 따로
검증하는 첫 단계입니다. 게임 연동, 행동 선택, 보상 설계, 전체 뇌 시뮬레이션은
아직 없습니다. 이 규칙은 **공학적 가설**이며 실제 초파리 학습을 검증·재현했다는
뜻이 아닙니다. 실행 예제의 세 뉴런과 body ID는 모두 합성 데이터입니다.

## 실행

Python 3.10 이상과 NumPy만 사용합니다. 저장소 루트에서:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m fly_dcss.demo
```

단위 테스트와 데모는 네트워크, GPU, 실제 커넥톰 파일이 필요 없습니다.

## 상태와 학습

각 뉴런의 활동 `x`는 [0, 1], 각 연결의 강도 `w`는 [0, weight_max]입니다.
부호는 출발 뉴런의 `sign`에 고정합니다. 한 스텝의 순서는 다음과 같습니다.

1. 이전 활동과 이전 가중치로 연결별 입력을 합산합니다
2. `x_next = a*x + (1-a)*clip(external + signed_W*x, 0, 1)`
3. `e_next = b*e + (1-b)*x_pre*x_next_post`
4. `w_next = clip(w + learning_rate*dt*modulator_post*e_next, 0, weight_max)`

`a=exp(-dt/tau_state)`, `b=exp(-dt/tau_eligibility)`입니다. 활동·가중치·시간은
모두 무차원 모델 단위입니다. 실제 밀리초나 발화율로 해석하지 않습니다.
가중치는 같은 스텝의 활동 계산 이후 변경되어 다음 스텝부터 사용됩니다.
이는 발화 기반 LIF/STDP 구현이 아닌 작은 연속 활동 모델입니다.

`plastic` 불리언 마스크로 수정할 연결을 고릅니다. 선택하지 않은 연결은 흔적도
쌓지 않습니다. `modulator`는 스칼라(전체 공통) 또는 뉴런별 벡터(도착 뉴런별)
이며 외부에서 공급하는 도파민 유사 조절 신호입니다. 도파민 뉴런 자체를 모델링하지
않습니다. 양수는 강화, 음수는 약화를 유도합니다. `learning=False`는 가중치만
고정하고 상태와 흔적은 계속 갱신합니다. 흔적을 지우려면 새 `Circuit`를 만듭니다.
초기 강도는 `clip(graph.weight * weight_scale, 0, weight_max)`입니다.

데모는 0~3 스텝에 입력, 7 스텝에 지연된 양의 조절 신호를 주고 12 스텝을 실행합니다.
첫 두 연결만 학습합니다. 결과는 다음과 같습니다.

```text
state:           [0.015837, 0.028430, 0.026081]
initial_weights: [0.400000, 0.300000, 0.200000]
final_weights:   [0.407125, 0.300679, 0.200000]
```

이는 구현 동작 확인이며 학습 성능 향상이나 게임 성공의 증거가 아닙니다.

## 기존 변환 데이터 연결

[flybrain](https://github.com/jppaquet/flybrain)의 `convert.py`가 생성하는
`malecns_graph.npz` 형식을 재사용합니다. 기존 로더의 outgoing CSR 배열 해석을
최소 어댑터로 분리했으며, 무거운 메타데이터·Arrow·시뮬레이터 의존성은 가져오지
않았습니다. 출처와 라이선스는 [THIRD_PARTY.md](../THIRD_PARTY.md)를 참조하세요.

- `indptr`: 출발 뉴런별 CSR 행 포인터, 길이 N+1
- `indices`: 연결의 도착 뉴런 인덱스
- `weight`: 부호 없는 연결별 시냅스 수
- `sign`: 출발 뉴런별 +1 또는 -1
- `body`: 원본 뉴런 ID, 길이 N

이미 적법하게 확보한 변환 파일과 검증한 ID 목록이 있을 때만 사용합니다:

```python
from fly_dcss.circuit import Circuit, load_converted

# selected_body_ids must come from the matching dataset/metadata, not demo IDs.
graph = load_converted("/path/to/malecns_graph.npz", body_ids=selected_body_ids)
circuit = Circuit(graph)
```

선택 순서대로 뉴런을 재색인하고, 양 끝이 선택된 연결만 유지합니다. 외부 회로의
입력과 출력 경계 연결은 버리므로 원래 회로의 동작을 보존한다고 가정하지 않습니다.
상태/강도는 실험 설정에 따른 임의 스케일이며 시냅스 수 자체를 직접 학습하지 않습니다.
부호는 upstream의 추정값을 그대로 사용하며 수용체나 세포별 생리학을 검증하지 않습니다.

이 어댑터는 NPZ를 메모리에 읽은 뒤 부분 회로를 고릅니다. 선택한 회로가 작아도
원본 파일 읽기에는 전체 배열 크기에 비례하는 메모리가 필요합니다. 이번 작업에서는
실데이터를 다운로드·변환·실행하지 않았습니다. 테스트 중 임시로 만든 작은 NPZ만으로
키, 방향, 부호, 재색인, 잘못된 입력 거부를 확인했습니다.

## 검증 범위

`python -m unittest discover -s tests -v`: 17개 테스트

- 동기 상태 전파의 수식 결과, 억제 부호, 상태·가중치 상하한
- 학습 비활성화, 조절 신호 없음, 흔적 없음 대조군
- 잔여 활동을 제거한 뒤 지연된 조절 신호, 흔적의 지수 감쇠
- 연결 마스크, 도착 뉴런별 조절, 재현성, 인스턴스 독립성
- 합성 NPZ 로딩, 부분 회로 재색인, 연결 없는 회로, 유효성 검사

Python 3.12.14 / NumPy 2.3.5 환경에서 확인했습니다. 실제 커넥톰 호환성은
소스 형식과 합성 왕복 테스트 수준이며 실제 파일 검증은 남아 있습니다.
