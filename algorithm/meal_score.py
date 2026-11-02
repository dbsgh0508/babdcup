"""
밥드컵 그룹 메뉴 추천 점수 알고리즘 v0.1
- 딥러닝 없이 규칙 기반으로 시작하고, 쌓인 데이터로 나중에 가중치를 학습시키는 구조
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────
# 1. 데이터 구조
# ─────────────────────────────────────────────
@dataclass
class Menu:
    name: str
    tags: set
    price: int  # 1인 평균 가격(원)

@dataclass
class Person:
    name: str
    pref: dict = field(default_factory=dict)      # 월드컵 결과 → 메뉴별 선호 점수 0~1
    hard_no: set = field(default_factory=set)     # 못 먹는 태그 (알레르기, 해산물 등) → 즉시 탈락
    dislikes: set = field(default_factory=set)    # 싫어하는 태그 → 감점
    budget: int | None = None                     # 1인 최대 예산


# 월드컵에서 몇 강까지 올라갔는지 → 선호 점수
ROUND_SCORE = {1: 1.0, 2: 0.85, 4: 0.7, 8: 0.55, 16: 0.4, 32: 0.25}

def worldcup_to_pref(results: dict) -> dict:
    """results = {"돈까스": 1(우승), "초밥": 2(결승), "라멘": 4(4강) ...}"""
    return {menu: ROUND_SCORE.get(r, 0.2) for menu, r in results.items()}


def load_menus(path="menus.json"):
    """menus.json을 읽어서 {메뉴명: Menu} 반환. 태그 오타가 있으면 바로 에러."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = {t for group in data["tag_groups"].values() for t in group}
    menus = {}
    for item in data["menus"]:
        unknown = set(item["tags"]) - vocab
        if unknown:
            raise ValueError(f"{item['name']}: 태그 목록에 없는 태그 {unknown}")
        if item["name"] in menus:
            raise ValueError(f"메뉴 이름 중복: {item['name']}")
        menus[item["name"]] = Menu(item["name"], set(item["tags"]), item["price"])
    return menus, vocab


# ─────────────────────────────────────────────
# 2. 파라미터 (나중에 데이터로 튜닝할 값들)
# ─────────────────────────────────────────────
DISLIKE_PENALTY = 0.25   # 싫어요 태그 하나당 감점
BUDGET_PENALTY_CAP = 0.4 # 예산 초과 최대 감점
TAG_SHRINK = 0.7         # 안 해본 메뉴를 태그로 추정할 때 확신도 (1이면 100% 믿음)
LIKE_TH, UNHAPPY_TH = 0.7, 0.35
CONF_W = 0.15            # 월드컵에서 직접 본 사람 비율 반영 (추정만으로 뜬 메뉴 견제)
MIN_SHOW = 50            # 이 점수 미만은 추천 목록에서 제외

# 모드별 α: 그룹점수 = α·평균 + (1-α)·최저점
#   α가 낮을수록 "한 명이라도 불만이면 손해" → 평화 모드
MODES = {
    "평화":   {"alpha": 0.4, "price_w": 0.0},
    "만족도": {"alpha": 0.8, "price_w": 0.0},
    "가성비": {"alpha": 0.6, "price_w": 0.25},
}


# ─────────────────────────────────────────────
# 3. 개인 점수
# ─────────────────────────────────────────────
# 취향이 아니라 '상황'을 나타내는 태그 → 취향 추정에서 제외
CONTEXT_TAGS = {"술자리", "혼밥", "단체", "밥", "면", "빵"}

def tag_affinity(p: Person, menus: dict) -> dict:
    """월드컵에서 고른/탈락시킨 메뉴들의 태그로 '태그 취향'을 역산"""
    s, c = {}, {}
    for name, score in p.pref.items():
        for t in menus[name].tags - CONTEXT_TAGS:
            s[t] = s.get(t, 0) + score
            c[t] = c.get(t, 0) + 1
    return {t: s[t] / c[t] for t in s}

def person_score(p: Person, m: Menu, aff: dict):
    # (1) 못 먹는 거 → 탈락 (None)
    if m.tags & p.hard_no:
        return None

    # (2) 기본 선호: 월드컵에서 직접 본 메뉴면 그 점수, 아니면 태그로 추정, 그것도 없으면 0.5(중립)
    if m.name in p.pref:
        base = p.pref[m.name]
    else:
        vals = [aff[t] for t in m.tags - CONTEXT_TAGS if t in aff]
        base = 0.5 + TAG_SHRINK * (sum(vals) / len(vals) - 0.5) if vals else 0.5

    # (3) 싫어요 감점
    base -= DISLIKE_PENALTY * len(m.tags & p.dislikes)

    # (4) 예산 초과 감점 (10% 초과 → -0.1)
    if p.budget and m.price > p.budget:
        base -= min(BUDGET_PENALTY_CAP, (m.price - p.budget) / p.budget)

    return max(0.0, min(1.0, base))


# ─────────────────────────────────────────────
# 4. 그룹 점수 + 추천 이유
# ─────────────────────────────────────────────
def recommend(people, menus: dict, mode="평화", top=3):
    cfg = MODES[mode]
    affs = {p.name: tag_affinity(p, menus) for p in people}
    prices = [m.price for m in menus.values()]
    pmin, pmax = min(prices), max(prices)

    ranked, excluded = [], []
    for m in menus.values():
        scores = {p.name: person_score(p, m, affs[p.name]) for p in people}
        blockers = [n for n, s in scores.items() if s is None]
        if blockers:
            excluded.append((m.name, blockers))
            continue

        vals = list(scores.values())
        mean, low = sum(vals) / len(vals), min(vals)
        g = cfg["alpha"] * mean + (1 - cfg["alpha"]) * low

        if cfg["price_w"]:
            cheap = 1 - (m.price - pmin) / (pmax - pmin) if pmax > pmin else 1
            g = (1 - cfg["price_w"]) * g + cfg["price_w"] * cheap

        # 직접 고른 사람이 많을수록 신뢰도 ↑
        direct = [p.name for p in people if m.name in p.pref]
        g *= (1 - CONF_W) + CONF_W * len(direct) / len(people)

        likers  = [n for n, s in scores.items() if s >= LIKE_TH]
        unhappy = [n for n, s in scores.items() if s < UNHAPPY_TH]
        reason = f"{len(likers)}명 선호" if direct else "취향 기반 추정"
        reason += f", 불만 {len(unhappy)}명({', '.join(unhappy)})" if unhappy else ", 불만 0명"
        notes = [f"{p.name} {'/'.join(sorted(m.tags & p.dislikes))} 비선호"
                 for p in people if m.tags & p.dislikes]
        if notes:
            reason += f" ({'; '.join(notes)})"
        if all(p.budget is None or m.price <= p.budget for p in people):
            reason += ", 전원 예산 OK"

        score = round(g * 100)
        if score >= MIN_SHOW:
            ranked.append((score, m.name, reason, scores))

    ranked.sort(reverse=True)
    return ranked[:top], excluded


# ─────────────────────────────────────────────
# 5. 데모: PPT 예시 (A~D 4명)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    menus, vocab = load_menus(Path(__file__).with_name("menus.json"))
    print(f"메뉴 {len(menus)}개, 태그 {len(vocab)}개 로드")

    # 16강 월드컵 (전원 같은 16개 메뉴). 우승→결승→4강(2)→8강(4)→16강 탈락(8)
    BRACKET = ["돈까스", "초밥", "라멘", "국밥", "삼겹살", "파스타", "떡볶이", "치킨",
               "김밥", "짜장면", "마라탕", "샐러드", "피자", "쌀국수", "제육볶음", "회"]

    def wc(champ, final, semi, quarter):
        r = {m: 16 for m in BRACKET}
        r.update({m: 8 for m in quarter})
        r.update({m: 4 for m in semi})
        r[final], r[champ] = 2, 1
        return worldcup_to_pref(r)

    people = [
        Person("A", wc("초밥", "돈까스", ["라멘", "회"], ["쌀국수", "파스타", "짜장면", "피자"]),
               hard_no={"매운맛"}),
        Person("B", wc("국밥", "삼겹살", ["돈까스", "제육볶음"], ["치킨", "짜장면", "김밥", "라멘"]),
               dislikes={"채소위주"}),
        Person("C", wc("파스타", "돈까스", ["피자", "샐러드"], ["초밥", "쌀국수", "치킨", "김밥"]),
               dislikes={"국물"}),
        Person("D", wc("떡볶이", "치킨", ["돈까스", "김밥"], ["짜장면", "제육볶음", "마라탕", "피자"]),
               budget=13000),
    ]
    for p in people:
        bad = (p.hard_no | p.dislikes) - vocab
        if bad:
            raise ValueError(f"{p.name}: 없는 태그 {bad}")

    for mode in MODES:
        top, excl = recommend(people, menus, mode)
        print(f"\n[{mode} 모드]")
        for i, (score, name, reason, _) in enumerate(top, 1):
            print(f"  {i}위 {name:5s} {score}점 — {reason}")
    print(f"\n제외 {len(excl)}개:", ", ".join(f"{n}({'/'.join(b)})" for n, b in excl))
