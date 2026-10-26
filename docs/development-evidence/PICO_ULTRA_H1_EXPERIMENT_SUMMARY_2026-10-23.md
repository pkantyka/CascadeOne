# CascadeOne — podsumowanie eksperymentu H1: Pico → Ultra (confidence)

**Status: H1 zakończony na poziomie jakości Gold i wydajności *live* przy pojedynczym kliencie.**
Data analizy: 2026-10-23. Eksperymenty: 2026-10-22.
Gałąź: `experiment/pico-ultra-live-perf-20261022`.

## 1. Cel i badane hipotezy

H1: Basal Omni Pico może odpowiadać samodzielnie na część pytań, a na pozostałych wywoływać Ultra, ograniczając średnie opóźnienie względem Ultra-only przy kontrolowanej stracie jakości. Należy zmierzyć, **ile czasu zyskujemy kosztem ilu punktów procentowych jakości względem Ultra**, nie względem idealnych 100%.

Eksperyment H2 (router przewidujący zysk z eskalacji na stanach ukrytych Pico) **nie jest częścią tych wyników**; jego przygotowanie znajduje się na gałęzi `experiment/pico-ultra-latent-router-20261022`.

## 2. Piny i protokół

- **Pico:** `Remek/basal-omni-pico`, revision `b31750af3ef1b45adfa765f73edbb84b5362983a`, Basal v2.3.2, commit silnika `776fde6c7d50a37b1dc1cce2ec1219c056ef2d76`.
- **Ultra:** `Remek/basal-omni-1.0-ultra`, revision `b8e214cb42d0a7a4cb2c3a93b0ba02c65d7590e4`, Basal v2.1.0, commit silnika `62542d31e8eb69000839c01e8f9b45da1fb80e49`.
- Obydwa: tryb `eager`, `orders=1`, `prompt_format=2`.
- **Reguła:** każdy request najpierw trafia do Pico. Jeśli `min(answers[qid].confidence) < threshold`, **cały request** jest ponownie przetwarzany przez Ultra; w przeciwnym razie zwracamy wynik Pico. Wariant `ultra_only` wywołuje wyłącznie Ultra.
- Progi ustalono **wyłącznie na calibration**; na ID/OOD nie były korygowane. Jednostka eskalacji = request, jednostka jakości = pytanie z oczekiwaną zgodnością z oryginalnym Gold (probability of predicted option; bez udawania, że rozkłady soft są twardą accuracy).
- **Jakość:** sparowana ewaluacja offline na kompletnym Gold: calibration 900 requestów / 2090 pytań; test ID 888 / 2050; test OOD-family 899 / 1957; historical 4969 / 15522 (historical jest eksploracyjny, nie niezależny held-out). Z oceny głównej wyłączono wcześniej oceniane pozycje i niekompletne Gold.
- **Czas:** rzeczywiste HTTP do modeli na DGX Spark GB10, 64 requesty ID + 64 OOD, 2 powtórzenia. Faza `ultra-alone` (Pico niezainicjalizowany) i `colocated` (Pico i Ultra równocześnie rezydują w pamięci GPU). Każdy wariant wykonywał realną inferencję, bez korzystania z cache odpowiedzi. Kolejność wariantów losowa, 8 warmupów na aktywny serwer; jeden sekwencyjny klient. Jakość i czas były mierzone na **różnych próbkach**, pochodzących z tych samych zamrożonych splitów.

## 3. Główny wynik: koszt jakości *względem Ultra* oraz oszczędność czasu

Baza dla jakości: **Ultra-only = 91,19% ID oraz 91,50% OOD** (wartości zaokrąglone).
Baza dla czasu: **Ultra-only colocated = 1646,9 ms średnio, p95 = 4526,4 ms**.

| Reguła (confidence Pico < próg) | Eskalacja w próbie czasu | Mean [ms] | Oszczędność mean vs Ultra | p95 [ms] | Zmiana p95 vs Ultra | Jakość ID (strata vs Ultra) | Jakość OOD (strata vs Ultra) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ultra-only colocated | 100% | 1646,9 | 0,0% | 4526,4 | 0,0% | 91,19% (**0,00 pp**) | 91,50% (**0,00 pp**) |
| Pico-only (nigdy Ultra) | 0% | 111,0 | **93,3%** | 302,3 | −93,3% | 66,97% (**−24,23 pp**) | 65,10% (**−26,41 pp**) |
| < 0,370892 | 5,5% | 231,1 | **86,0%** | 695,9 | −84,6% | 70,58% (**−20,62 pp**) | 68,27% (**−23,24 pp**) |
| < 0,508680 | 19,5% | 786,0 | **52,3%** | 4829,1 | +6,7% | 76,63% (**−14,57 pp**) | 76,85% (**−14,65 pp**) |
| < 0,656352 | 45,3% | 1237,3 | **24,9%** | 4820,7 | +6,5% | 84,14% (**−7,05 pp**) | 84,68% (**−6,82 pp**) |
| < 0,856891 | 75,8% | 1599,1 | **2,9%** | 4837,7 | +6,9% | 89,72% (**−1,47 pp**) | 90,21% (**−1,30 pp**) |
| Pico → Ultra zawsze | 100% | 1758,1 | **−6,8%** | 4837,0 | +6,9% | 91,19% (**0,00 pp**) | 91,50% (**0,00 pp**) |

**Ultra-alone** (osobna faza, bez załadowanego Pico): mean **1648,2 ms**, p50 **832,5 ms**, p95 **4540,0 ms**, p99 **8103,8 ms**. Ultra-only colocated: p50 **812,7 ms**, p99 **8119,9 ms**. Różnica średniej pomiędzy samym Ultra a współrezydującym Ultra to **−0,08%**; nie wykazano narzutu współrezydencji w badanym sekwencyjnym scenariuszu.

**Uwaga o eskalacji:** 5,5%, 19,5%, 45,3%, 75,8% to odsetki z 128-requestowej próbki czasu. Na pełnym Gold ID/OOD odpowiednio: 7,9%/6,9%; 23,9%/23,9%; 49,9%/47,4%; 75,8%/71,6%. Progi są identyczne, ale rozkłady i liczności nie są.

## 4. Niepewność i test integralności

Dla **oszczędności średniego czasu** względem Ultra-only colocated przedziały 95% (bootstrap parowany po ID zapytania, 600 replikacji):

| Reguła | Oszczędność mean | 95% CI |
|---|---:|---:|
| Pico-only | +93,3% | [+93,0%; +93,6%] |
| < 0,370892 | +86,0% | [+78,5%; +91,8%] |
| < 0,508680 | +52,3% | [+38,6%; +67,6%] |
| < 0,656352 | +24,9% | [+15,5%; +35,9%] |
| < 0,856891 | +2,9% | [−0,9%; +8,3%] |
| Pico → zawsze Ultra | −6,8% | [−7,1%; −6,4%] |

Dla p95 **nie wyliczono przedziałów ufności**; wzrost p95 w tabeli to obserwacja, nie wykazana istotność statystyczna.

Zarchiwizowano i zweryfikowano: **256/256** prób `ultra-alone` i **1792/1792** prób `colocated`, czyli 7 wariantów × 128 requestów × 2 powtórzenia. Nie ma duplikatów kluczy `(repeat, id, arm)`, brak brakujących pomiarów; we wszystkich rekordach reguła eskalacji odpowiada zapisanemu confidence. Piny i SHA256 próby oraz progów są zgodne z handoffem. Dane dowodowe i `SHA256SUMS`: [`reports/pico_ultra_h1_20261022/`](../reports/pico_ultra_h1_20261022/).

## 5. Ocena wyników i decyzja

**H1: potwierdzona dla oszczędności średniego opóźnienia, niepotwierdzona jako rozwiązanie o jakości Ultra przy zbliżonym koszcie.** Confidence Pico jest użytecznym sygnałem routingu, ale nie odzyskuje dość jakości przy niskim koszcie eskalacji. Przy progu 0,656352 mean maleje o **409,6 ms (24,9%)**, lecz jakość ID spada względem Ultra o **7,05 pp**, OOD o **6,82 pp**, a p95 jest wyższe o **6,5%**. To najlepszy zbadany kompromis do porównania z routerem H2, **nie rekomendacja produkcyjnego wdrożenia**.

Przy progu 0,856891 jakość zbliża się do Ultra (strata tylko 1,47 / 1,30 pp), ale korzyść ze średniego czasu to zaledwie **2,9%**, z 95% CI obejmującym zero. Przy progu 0,508680 oszczędzamy **52,3%** średniej, lecz tracimy **14,57 / 14,65 pp** jakości. Wariant Pico → zawsze Ultra dowodzi kosztu dodatkowego etapu: **+6,8%** średniego czasu.

**Wyjaśnienie p95:** trudniejsze przypadki często wymagają eskalacji; dla progu 0,656352 wywołania Ultra na tej podgrupie trwają średnio ok. **2487 ms** (a przy 0,508680 ok. **3458 ms**) wobec **1647 ms** na całej próbie. Do eskalowanego requestu dochodzi jeszcze ok. **111 ms** Pico. Dlatego średnia może spadać, a p95 wzrastać.

**Brak produkcyjnego GO:** nie badano równoległych klientów, stabilnego throughput przy obciążeniu, wykorzystania VRAM/GPU, energii, cold startu, realnego transportu między maszynami, UDP ani p95 CI. Zgodność checkpointów wynika z pinów manifestu/startupu i tożsamości HTTP; nie jest kryptograficznym poświadczeniem wag w GPU.

**Następny etap H2:** nauczyć router latentnego z funkcją celu **Gold(Ultra) − Gold(Pico)**, porównać z confidence-only przy **tym samym budżecie eskalacji** na ID i OOD, następnie zmierzyć jego realny dodatkowy koszt inferencji. Kryterium porównawcze: dla ~50% eskalacji przebić **84,14% ID / 84,68% OOD** przy nie większym koszcie niż **1237 ms mean / 4821 ms p95**; wartości czasu są z osobnej próby i wymagają pomiaru H2 na identycznych requestach.

## 6. Pliki, replay i łańcuch dowodowy

Zestaw dowodowy zawiera surowe `ultra-alone.jsonl` i `colocated.jsonl`, odpowiadające im podsumowania, `confidence.json` (cały wynik jakości H1), `sample.jsonl` (zamrożona próba opóźnień), `manifest.json` (piny/model/SHAs) oraz `SHA256SUMS`. Skopiowane dane są identyczne bajtowo z otrzymanymi wynikami; oryginalny katalog `artifacts/pico-cascade-20261022/perf_results/output/` nie został zmodyfikowany.

Oryginalny plan: [`PICO_ULTRA_CASCADE_EXPERIMENT_2026-10-22.md`](PICO_ULTRA_CASCADE_EXPERIMENT_2026-10-22.md). Handoff live: [`PICO_ULTRA_LIVE_PERF_HANDOFF_2026-10-22.md`](PICO_ULTRA_LIVE_PERF_HANDOFF_2026-10-22.md). Szczegółowa wcześniejsza analiza czasów: [`PICO_ULTRA_LIVE_PERF_RESULTS_2026-10-23.md`](PICO_ULTRA_LIVE_PERF_RESULTS_2026-10-23.md). Skrypty: `experiments/pico_cascade/analyze.py` i `experiments/pico_cascade/perf/benchmark.py`.
