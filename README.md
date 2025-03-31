# 상공회의소 경기인력개발원 인텔교육 3기

## Clone code 

```shell
git clone --recurse-submodules https://github.com/kccistc/intel-03.git
```

* `--recurse-submodules` option 없이 clone 한 경우, 아래를 통해 submodule update

```shell
git submodule update --init --recursive
```

## Preparation

### Git LFS(Large File System)

* 크기가 큰 바이너리 파일들은 LFS로 관리됩니다.

* git-lfs 설치 전

```shell
# Note bin size is 132 bytes before LFS pull

$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

* git-lfs 설치 후, 다음의 명령어로 전체를 가져 올 수 있습니다.

```shell
$ sudo apt install git-lfs

$ git lfs pull
$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

### 환경설정

* [Ubuntu](./doc/environment/ubuntu.md)
* [OpenVINO](./doc/environment/openvino.md)
* [OTX](./doc/environment/otx.md)

## Team projects

### 제출방법

1. 팀구성 및 프로젝트 세부 논의 후, 각 팀은 프로젝트 진행을 위한 Github repository 생성

2. [doc/project/README.md](./doc/project/README.md)을 각 팀이 생성한 repository의 main README.md로 복사 후 팀 프로젝트에 맞게 수정 활용

3. 과제 제출시 `인텔교육 3기 Github repository`에 `New Issue` 생성. 생성된 Issue에 하기 내용 포함되어야 함.

    * Team name : Project Name
    * Project 소개
    * 팀원 및 팀원 역활
    * Project Github repository
    * Project 발표자료 업로드

4. 강사가 생성한 `Milestone`에 생성된 Issue에 추가 

### 평가방법

* [assessment-criteria.pdf](./doc/project/assessment-criteria.pdf) 참고

### 제출현황

### Team: 뭔가 센스있는 팀명
<프로젝트 요약>
* Members
  | Name | Role |
  |----|----|
  | 채치수 | Project lead, 프로젝트를 총괄하고 망하면 책임진다. |
  | 송태섭 | Project manager, 마일스톤을 생성하고 프로젝트 이슈 진행상황을 관리한다. |
  | 정대만 | UI design, 사용자 인터페이스를 정의하고 구현한다. |
  | 채소연 | AI modeling, 원하는 결과가 나오도록 AI model을 선택, data 수집, training을 수행한다. |
  | 권준호 | Architect, 프로젝트의 component를 구성하고 상위 디자인을 책임진다. |
* Project Github : https://github.com/goodsense/project_awesome.git
* 발표자료 : https://github.com/goodsense/project_aewsome/doc/slide.ppt

## Team name : 약손
<프로젝트 요약>
시각장애인 분들이 약품을 섭취하는데 있어, 약 위치를 기억하는 것과, 약의 내용을 파악하는데 어려움을 겪는바,
이 문제를 해소하고 자 함

* Members
  | Name | Role |
  |----|----|
  | 이동준 | Project manager, 프로젝트를 총괄, 코드 병합 |
  | 박원석 | Project manager, 마일스톤을 생성하고 프로젝트 이슈 진행상황을 관리한다. |
  | 김건휘 | UI design, 사용자 인터페이스를 정의하고 구현한다. |
  | 강동수 | AI modeling, 원하는 결과가 나오도록 AI model을 선택, data 수집, training을 수행한다. |
  | 윤승건 | Architect, 프로젝트의 component를 구성하고 상위 디자인을 책임진다. |

* Project Github : https://github.com/cschuadj721/SigakDoumi

* 발표자료 : https://github.com/cschuadj721/SigakDoumi/blob/master/%EC%8B%9C%EA%B0%81%EC%9E%A5%EC%95%A0%EC%9D%B8%20%EA%B2%BD%EA%B5%AC%EC%95%BD%20%EC%84%AD%EC%B7%A8%20%26%20%EB%AC%BC%ED%92%88%EB%B3%B4%EA%B4%80%20%EB%8F%84%EC%9A%B0%EB%AF%B8(%EC%88%98%EC%A0%95%EB%B3%B8).pptx


## Team name : 재활로봇(팀MAX)
<프로젝트 요약>
기존 PCM (CKC) 다리 재활 운동 기구의 한계를 개선하여 AC 모터 기반의 액티브(Active) 및 패시브(Passive) 모드를 지원하는 스마트 재활 기구 개발. AI 모방학습(BC) 와 강화학습(RL)을 활용해 사용자 맞춤형 운동 보조와 데이터 기반 재활 관리를 제공

* Members
  | Name | Role |
  |----|----|
  | 안세환 | Project manager, 프로젝트를 총괄, 코드 병합 |
  | 송선대 | Project manager, 마일스톤을 생성하고 프로젝트 이슈 진행상황을 관리한다. |
  | 김관우 | UI design, 사용자 인터페이스를 정의하고 구현한다. |
  | 김정헌 | AI modeling, 원하는 결과가 나오도록 AI model을 선택, data 수집, training을 수행한다. |
  | 최은택 | Architect, 프로젝트의 component를 구성하고 상위 디자인을 책임진다. |

* Project Github : https://github.com/Sehwani2/AI_CPMRehab 
* 발표자료 : https://intel03-team02.my.canva.site/

## Team name : 잘걷는 도비

신발에 어레이 FSR 센서를 이용해 동적 족저압을 측정하여 CRNN 레이어를 활용한 딥러닝 보행이상 검출 신발 제작 프로젝트

* Members
  | Name | Role |
  |----|----|
  | 정지민 | Project lead, 프로젝트를 총괄, 원하는 결과가 나오도록 AI model을 선택, data 수집, training을 수행한다. |
  | 이현종 | Lead Embedded Software Engineer, 임베디드 기기 설계 및 구현, 센서 튜닝 등 임베디드 기기 총괄한다. |
  | 이효원 | AI modeling, Embedded Engineer, 임베디드 기기 설계, data 수집, training을 수행한다. |
  | 조재상 | Project manager, UI design, 프로젝트 Git릏 관리하고, 사용자 인터페이스를 구현한다. |
* Project Github : https://github.com/jo5862/intel03_team3_foot 
* 발표자료 : https://github.com/jo5862/intel03_team3_foot/blob/main/Ppt/team03.pdf


