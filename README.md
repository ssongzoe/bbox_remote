# ACT 원격 추론

로봇 PC는 CPU로 CSRT·BB 변환과 action 실행을 담당하고 GPU PC는 ACT를 담당합니다.
두 PC에서 동일한 수정 LeRobot 버전을 사용하세요. LeRobot 소스 수정은 필요 없습니다.
로봇 PC에도 LeRobot/CPU PyTorch와 GUI/CSRT 지원 OpenCV는 필요합니다.
TCP 7447을 사용하며 예전 gRPC :8080 서버와 연결되지 않습니다.

## original: GPU PC에서 서버 실행

체크포인트와 데이터 경로를 실제 경로로 바꾸세요. 두 서버는 같은 포트를 쓰므로 하나씩 실행합니다.

```bash
python act_bb_remote_server.py \
  --checkpoint "/실제/original/checkpoints/030000/pretrained_model" \
  --dataset-root "/실제/dataset/pink_all" \
  --deployment pink_original \
  --listen tcp/0.0.0.0:7447
```

## original: 로봇 PC에서 클라이언트 실행

GPU PC IP는 기존 192.168.0.41을 사용했습니다. 서버 준비 후 실행하세요.

```bash
QT_QPA_PLATFORM=xcb python -u act_bb_remote_client.py \
  --bb-mode original \
  --bb-source-key left \
  --inference.type=remote \
  --inference.endpoint=tcp/192.168.0.41:7447 \
  --inference.deployment=pink_original \
  --inference.semantics=rby1_left_joint_v1 \
  --inference.mode=chunk \
  --inference.chunk_merge=aligned \
  --inference.refill_seconds=0.5 \
  --inference.encoding=raw \
  --fps=30 \
  --robot.type=rby1 \
  --robot.address=192.168.30.1:50051 \
  --robot.model=m \
  --robot.action_mode=joint \
  --robot.use_left_arm=true \
  --robot.use_right_arm=false \
  --robot.use_torso=false \
  --robot.use_mobile_base=false \
  --robot.use_gripper=true \
  --robot.use_velocity=false \
  --robot.use_torque=false
```

## bbox: GPU PC에서 서버 실행

체크포인트와 데이터 경로를 실제 경로로 바꾸세요. 두 서버는 같은 포트를 쓰므로 하나씩 실행합니다.

```bash
python act_bb_remote_server.py \
  --checkpoint "/실제/bbox/checkpoints/030000/pretrained_model" \
  --dataset-root "/실제/dataset/pink_all_bb" \
  --deployment pink_bbox \
  --listen tcp/0.0.0.0:7447
```

## bbox: 로봇 PC에서 클라이언트 실행

GPU PC IP는 기존 192.168.0.41을 사용했습니다. 서버 준비 후 실행하세요.

```bash
QT_QPA_PLATFORM=xcb python -u act_bb_remote_client.py \
  --bb-mode bbox \
  --bb-source-key left \
  --bb-score-threshold 0.65 \
  --bb-score-margin 0.01 \
  --bb-target-missing-frames 5 \
  --inference.type=remote \
  --inference.endpoint=tcp/192.168.0.41:7447 \
  --inference.deployment=pink_bbox \
  --inference.semantics=rby1_left_joint_v1 \
  --inference.mode=chunk \
  --inference.chunk_merge=aligned \
  --inference.refill_seconds=0.5 \
  --inference.encoding=raw \
  --fps=30 \
  --robot.type=rby1 \
  --robot.address=192.168.30.1:50051 \
  --robot.model=m \
  --robot.action_mode=joint \
  --robot.use_left_arm=true \
  --robot.use_right_arm=false \
  --robot.use_torso=false \
  --robot.use_mobile_base=false \
  --robot.use_gripper=true \
  --robot.use_velocity=false \
  --robot.use_torque=false
```

## 사용법

- 시작 자세에서 object, gripper, target(검은 고리)을 선택합니다. 기본적으로 gripper ROI만 저장해 재사용합니다.
- 타겟은 초반 숨김 → 후보가 5개 처리 이미지 연속 없으면 사라짐 판정 → 재검출 시 자동 입력 활성화입니다. T는 예외 상황 수동 전환용입니다.
- 타겟 출력에만 좌우 20%, 위아래 10%, 최소 15px 여백을 적용합니다.
- object/gripper 추적 실패는 중단하고 target 실패는 재검색합니다.
- S 성공 종료, F 실패 종료, Q/Esc 중단. 한 실행이 한 trial입니다.
- 첫 연결 확인은 --bb-dry-run을 추가하세요. 정책 action은 전송하지 않지만 드라이버 연결/초기화/종료 처리는 실행됩니다.
- 원본/BB를 로봇에 동시에 연결하지 마세요.
- 원본/BB 모두 같은 aligned chunk 병합, 50 action slice, refill 0.5초를 사용합니다. 예전 weighted_average와 동일한 알고리즘은 아닙니다.
- CPU 추적은 최신 프레임을 처리하고 지연 시 중간 프레임을 건너뜁니다. 처리 시간은 tracking.jsonl에 기록됩니다.
- 학습 meta/info.json의 action/state 이름과 순서로 서버 계약을 만듭니다. 실제 로봇 플러그인 이름이 다르면 handshake 오류가 납니다. 임의 이름/순서 변환은 하지 않습니다.
- 로봇 PC에는 체크포인트, --policy.path, CUDA 설정을 넣지 않습니다.
- 구문 검사와 실제 CSRT 합성 영상 워커·자동 활성화 테스트는 통과했습니다. 실제 GPU 모델 로드, Zenoh 왕복과 로봇 연결은 사용자 장비에서 확인해야 합니다.

