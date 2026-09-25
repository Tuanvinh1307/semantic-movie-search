# Bao cao danh gia mo hinh truy hoi phim

## Cau hinh danh gia
- So query: 200
- Top-N danh gia: 10
- Pipeline: AdaptiveSearchPipeline (Query -> Hybrid -> RRF -> Aggregate -> Router -> HyDE/Rerank -> Final)
- File ket qua chi tiet: `D:\doan\Đồ Án Truy Hồi Thông Tin\semantic-movie-search-master\evaluation\evaluation_results.csv`

## Ket qua tong quan
- Hit@1: 0.8500
- Hit@3: 0.8900
- Hit@5: 0.9000
- Hit@10: 0.9000
- MRR@10: 0.8722
- Latency trung binh/query: 2.907s
- Latency p50: 3.695s
- Latency p95: 4.189s
- Route EASY: 4 (2.0%)
- Route HARD: 196 (98.0%)

## Ket qua theo nhom query
- cast_director: Hit@1=0.8600, Hit@3=0.9600, Hit@5=0.9600, Hit@10=0.9600, MRR=0.9100
- keyword_theme: Hit@1=0.5800, Hit@3=0.6200, Hit@5=0.6600, Hit@10=0.6600, MRR=0.6090
- plot_memory: Hit@1=1.0000, Hit@3=1.0000, Hit@5=1.0000, Hit@10=1.0000, MRR=1.0000
- year_constraint: Hit@1=0.9600, Hit@3=0.9800, Hit@5=0.9800, Hit@10=0.9800, MRR=0.9700

## Mau truy van that bai (khong tim thay dung trong Top-10)
- q3: `movie about highway and sunscreen in a drama setting` | expected: `Home` | top1: `Lost Highway` | route: HARD
- q11: `movie about hustler and male prostitution in a drama setting` | expected: `The Last Match` | top1: `Sauvage` | route: HARD
- q23: `movie about dreams and paris in a mystery setting` | expected: `Caché` | top1: `The Science of Sleep` | route: HARD
- q27: `movie about daughter and individual in a comedy setting` | expected: `Muriel's Wedding` | top1: `Freaky Friday` | route: HARD
- q31: `movie about usa president and nuclear war in a mystery setting` | expected: `Watchmen` | top1: `Thirteen Days` | route: HARD
- q38: `find a action movie directed by Tony Scott with Denzel Washington and Chris Pine` | expected: `Unstoppable` | top1: `Man on Fire` | route: HARD
- q39: `movie about pennsylvania and usa in a action setting` | expected: `Unstoppable` | top1: `Shooter` | route: HARD
- q50: `find a drama movie directed by Céline Sciamma with Zoé Héran and Malonn Lévana` | expected: `Tomboy` | top1: `L'Emprise` | route: HARD
- q71: `movie about jealousy and parent child relationship in a comedy setting` | expected: `Bridesmaids` | top1: `The Parent Trap` | route: HARD
- q79: `movie about work and sibling relationship in a animation setting` | expected: `Ratatouille` | top1: `Brother Bear` | route: HARD
- q87: `movie about high school and martial arts in a animation setting` | expected: `Kim Possible Movie: So the Drama` | top1: `TEKKEN: Blood Vengeance` | route: HARD
- q99: `movie about drama and drama in a drama setting` | expected: `Honey` | top1: `Gods` | route: HARD
- q103: `movie about civil war and war crimes in a war setting` | expected: `Tears of the Sun` | top1: `Civil War` | route: HARD
- q104: `war film released around 2003 with plot like navy seal lieutenant a.k. waters and his elite squadron of tactical specialists are forced to choose between their` | expected: `Tears of the Sun` | top1: `Warfare` | route: HARD
- q143: `movie about nurse and alcohol in a drama setting` | expected: `Arrhythmia` | top1: `The Good Nurse` | route: HARD

## De xuat cai tien
- Bo sung tap query hard bang tay (memory-based, slang, typo) de giam do lech do du lieu sinh tu template.
- Tach benchmark theo nam/the loai de nhin duoc domain nao yeu.
- Them metric nDCG@10 va Recall@50 cho lop retrieval truoc rerank.
- Them so sanh A/B: co-HyDE va khong-HyDE de luong hoa tac dong cua nhanh HARD.
- Thiet lap bo test hoi quy (regression set) 50 query co dinh sau moi lan thay doi pipeline.
