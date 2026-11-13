# VS Code · WSL · GitHub 공유 안내

## 점검 당시 상태 — 2026-11-13

- 프로젝트: `/home/archant/projects/MultiAgent_MetaRL_Project`, VS Code WSL 환경.
- Git 저장소가 이미 있고 현재 브랜치는 `master`다. `git init`이나 브랜치 강제 변경은 필요 없다.
- 최신 커밋은 `d17f436 Initial MAMRL project setup`이며 이후 코드·문서가 수정/미추적 상태다.
- `git remote -v`에 출력이 없으므로 현재 저장소에 원격은 설정되어 있지 않다.
- WSL의 `code` 명령은 있고 `gh` 명령은 찾지 못했다. VS Code 기본 Git 기능으로 게시할 수 있다.
- 이 정리 작업은 commit/push/저장소 생성/동료 초대를 실행하지 않았다.

## 먼저 전달할 자료

1. [동료 전달용 요약](sessions/08_colleague_progress_note.md)
2. [구현 상태와 실제 검증 기록](Implementation_Status.md)
3. [탐구 경과](Research_Log.md)
4. [아이디어 색인](../03_ideas/Idea_Inbox.md)과 [Generator 연구 노트](../03_ideas/Rule_Generator_Research_Notes.md)
5. [독립 검토와 다음 방향](sessions/07_independent_research_review.md)
6. [현재 구현·설치·독립 실행 방법](../README.md)

## VS Code에서 처음 게시하기

1. WSL 터미널에서 아래처럼 열고 VS Code 왼쪽 아래가 WSL 연결인지 확인한다.

   ```bash
   cd /home/archant/projects/MultiAgent_MetaRL_Project
   code .
   ```

2. Source Control(`Ctrl+Shift+G`)에서 변경·미추적 파일을 검토하고 공유할 코드/문서를 Stage한다.
   현재 구현 파일 상당수가 미추적이므로 기존 추적 파일만 commit하면 구현이 빠진다.
3. 메시지를 입력해 Commit한다. Git 이름/이메일을 요청하면 WSL에서 본인 정보를 설정한다.
   이름/이메일은 commit 작성자 정보이며 GitHub 로그인과는 별개다.
4. `Ctrl+Shift+P` → **Publish to GitHub**를 실행한다. 표시되지 않으면 Source Control의 게시 메뉴를 확인한다.
5. GitHub 로그인 안내를 완료하고 저장소 이름·Private/Public을 선택한다. 동료 협업 목적에는 Private으로 시작할 수 있다.
6. GitHub에서 파일과 최신 commit을 확인한다. 저장소 Settings의 Collaborators/접근 관리에서 동료를 초대한다.
   초대 수락 후 동료는 저장소를 clone할 수 있다.

기본 인증·commit·push에 GitHub CLI나 PR 확장이 필수는 아니다.
이후에는 commit 후 Push/Sync로 전송한다. 동료 작업과 겹칠 때는 각자 브랜치와 PR을 쓰는 편이 좋다.

## GitHub에서 빈 저장소를 먼저 만든 경우

VS Code 게시 방식 대신 사용할 대안이다. GitHub에서 README/.gitignore/license를 추가하지 않은
빈 저장소를 만들고 안내 URL을 사용한다. 아래는 사용자가 실행할 예시이며 이번에 실행하지 않았다.

```bash
git status
git add .
git diff --cached --stat
git commit -m "Add toy implementation, experiment records and research notes"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin HEAD
```

`git add .` 전에 공유 대상을 확인한다. 이후 remote가 이미 생겼다면 `remote add`를 반복하지 않는다.
인증은 VS Code 로그인이나 별도로 구성한 HTTPS/SSH 인증을 사용한다.

## 결과·재현성 범위

`.venv/`, `runs/`, `checkpoints/`, `.env` 등은 현재 `.gitignore`에 따라 제외된다.
동료는 가상환경을 새로 만들고 `requirements-training.txt`로 선택 학습 의존성을 설치해야 한다.
실행 명령은 루트 README를 따른다.

동료는 README의 명령으로 각자 독립 실행하고, 자기 `runs/`에서 원자료와 HTML을 생성한다.
이 공유 정리에는 실행 결과 사본을 포함하지 않는다. 연구적 관찰의 요약과 실제 실행 명령은 구분한다.

## 공식 안내

- [Microsoft: VS Code와 WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode)
- [VS Code: 저장소와 원격, Publish to GitHub](https://code.visualstudio.com/docs/sourcecontrol/repos-remotes)
- [VS Code: GitHub 인증과 협업](https://code.visualstudio.com/docs/sourcecontrol/github)
- [GitHub: 기존 로컬 코드 게시](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
