#!/usr/bin/env python3
"""
Cross-Platform Coherence Analysis
Analyzes cognitive events across ChatGPT and Claude platforms in the UCW database.
"""

import asyncio
import asyncpg
from datetime import datetime, timezone

DSN = "postgresql://localhost:5432/ucw_cognitive"

# ─── Formatting helpers ───────────────────────────────────────────────

def header(title, width=80):
    return f"\n{'━' * width}\n  {title}\n{'━' * width}"

def subheader(title):
    return f"\n  ── {title} {'─' * (74 - len(title))}"

def bar(value, max_value, width=40):
    filled = int((value / max_value) * width) if max_value else 0
    return '█' * filled + '░' * (width - filled)

def pct(part, total):
    return f"{(part/total*100):.1f}%" if total else "0.0%"

def trunc(text, length=80):
    if not text:
        return "(empty)"
    text = text.replace('\n', ' ').strip()
    return text[:length] + "..." if len(text) > length else text

# ─── Queries ──────────────────────────────────────────────────────────

async def run_analysis():
    conn = await asyncpg.connect(DSN)
    report = []

    report.append(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              UNIVERSAL COGNITIVE WALLET — COHERENCE ANALYSIS               ║
║                         Cross-Platform Intelligence Report                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'):>63s}  ║
║  Database:  ucw_cognitive (PostgreSQL + pgvector)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝""")

    # ═══════════════════════════════════════════════════════════════════
    # 1. PLATFORM DISTRIBUTION
    # ═══════════════════════════════════════════════════════════════════
    report.append(header("1. PLATFORM DISTRIBUTION"))

    # Total events per platform
    platform_events = await conn.fetch("""
        SELECT platform, COUNT(*) as event_count
        FROM cognitive_events
        GROUP BY platform
        ORDER BY event_count DESC
    """)

    total_events = sum(r['event_count'] for r in platform_events)
    report.append(subheader("Events by Platform"))
    report.append(f"  {'Platform':<25} {'Events':>10} {'Share':>8}  Distribution")
    report.append(f"  {'─'*25} {'─'*10} {'─'*8}  {'─'*40}")
    for r in platform_events:
        report.append(f"  {r['platform']:<25} {r['event_count']:>10,} {pct(r['event_count'], total_events):>8}  {bar(r['event_count'], total_events)}")
    report.append(f"  {'TOTAL':<25} {total_events:>10,}")

    # Sessions per platform
    platform_sessions = await conn.fetch("""
        SELECT platform, COUNT(DISTINCT session_id) as session_count
        FROM cognitive_events
        GROUP BY platform
        ORDER BY session_count DESC
    """)

    total_sessions = sum(r['session_count'] for r in platform_sessions)
    report.append(subheader("Sessions by Platform"))
    report.append(f"  {'Platform':<25} {'Sessions':>10} {'Share':>8}  {'Avg Events/Session':>20}")
    report.append(f"  {'─'*25} {'─'*10} {'─'*8}  {'─'*20}")
    for r in platform_sessions:
        ev_count = next((p['event_count'] for p in platform_events if p['platform'] == r['platform']), 0)
        avg_ev = ev_count / r['session_count'] if r['session_count'] else 0
        report.append(f"  {r['platform']:<25} {r['session_count']:>10,} {pct(r['session_count'], total_sessions):>8}  {avg_ev:>18.1f}")
    report.append(f"  {'TOTAL':<25} {total_sessions:>10,}")

    # Cognitive modes per platform
    mode_dist = await conn.fetch("""
        SELECT platform, cognitive_mode, COUNT(*) as cnt
        FROM cognitive_events
        WHERE cognitive_mode IS NOT NULL
        GROUP BY platform, cognitive_mode
        ORDER BY platform, cnt DESC
    """)

    report.append(subheader("Cognitive Modes by Platform"))
    current_platform = None
    for r in mode_dist:
        if r['platform'] != current_platform:
            current_platform = r['platform']
            plat_total = sum(x['cnt'] for x in mode_dist if x['platform'] == current_platform)
            report.append(f"\n  [{current_platform}] ({plat_total:,} events with mode)")
        report.append(f"    {r['cognitive_mode']:<20} {r['cnt']:>8,}  {pct(r['cnt'], plat_total):>7}  {bar(r['cnt'], plat_total, 30)}")

    # Embeddings coverage
    embed_stats = await conn.fetch("""
        SELECT ce.platform, COUNT(ec.source_event_id) as embedded, COUNT(ce.event_id) as total
        FROM cognitive_events ce
        LEFT JOIN embedding_cache ec ON ce.event_id = ec.source_event_id
        GROUP BY ce.platform
    """)

    report.append(subheader("Embedding Coverage"))
    for r in embed_stats:
        report.append(f"  {r['platform']:<25} {r['embedded']:>8,} / {r['total']:>8,} events embedded ({pct(r['embedded'], r['total'])})")
    total_embedded = sum(r['embedded'] for r in embed_stats)
    report.append(f"  {'TOTAL':<25} {total_embedded:>8,} / {total_events:>8,} events embedded ({pct(total_embedded, total_events)})")

    # ═══════════════════════════════════════════════════════════════════
    # 2. TOP CROSS-PLATFORM SEMANTIC MATCHES
    # ═══════════════════════════════════════════════════════════════════
    report.append(header("2. TOP CROSS-PLATFORM SEMANTIC MATCHES"))
    report.append("  Finding strongest semantic bridges between Claude and ChatGPT...\n")

    semantic_matches = await conn.fetch("""
        WITH claude_events AS (
            SELECT ec.embedding, ec.content_preview, ec.source_event_id, ce.cognitive_mode
            FROM embedding_cache ec
            JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
            WHERE ce.platform = 'claude-desktop'
            LIMIT 500
        )
        SELECT c.content_preview AS claude_text, c.source_event_id AS claude_id,
               c.cognitive_mode AS claude_mode,
               t.content_preview AS chatgpt_text, t.source_event_id AS chatgpt_id,
               t_ce.cognitive_mode AS chatgpt_mode,
               1 - (c.embedding <=> t.embedding) AS similarity
        FROM claude_events c
        CROSS JOIN LATERAL (
            SELECT ec2.content_preview, ec2.source_event_id, ec2.embedding
            FROM embedding_cache ec2
            JOIN cognitive_events ce2 ON ec2.source_event_id = ce2.event_id
            WHERE ce2.platform = 'chatgpt'
            ORDER BY ec2.embedding <=> c.embedding
            LIMIT 1
        ) t
        JOIN cognitive_events t_ce ON t.source_event_id = t_ce.event_id
        WHERE 1 - (c.embedding <=> t.embedding) >= 0.65
        ORDER BY similarity DESC
        LIMIT 30
    """)

    if semantic_matches:
        report.append(f"  Found {len(semantic_matches)} cross-platform matches (similarity >= 0.65)\n")

        # Summary statistics
        sims = [float(r['similarity']) for r in semantic_matches]
        report.append(f"  Similarity range: {min(sims):.4f} — {max(sims):.4f}")
        report.append(f"  Mean similarity:  {sum(sims)/len(sims):.4f}")
        report.append(f"  Median:           {sorted(sims)[len(sims)//2]:.4f}")

        # Similarity distribution
        report.append(subheader("Similarity Distribution"))
        brackets = [(0.90, 1.01, "0.90+  (near-identical)"),
                     (0.85, 0.90, "0.85-0.90 (very high)"),
                     (0.80, 0.85, "0.80-0.85 (high)"),
                     (0.75, 0.80, "0.75-0.80 (strong)"),
                     (0.70, 0.75, "0.70-0.75 (moderate)"),
                     (0.65, 0.70, "0.65-0.70 (threshold)")]
        for lo, hi, label in brackets:
            cnt = sum(1 for s in sims if lo <= s < hi)
            report.append(f"    {label:<30} {cnt:>4}  {bar(cnt, len(sims), 30)}")

        # Top 15 detailed matches
        report.append(subheader("Top 15 Matches (Detailed)"))
        for i, r in enumerate(semantic_matches[:15], 1):
            sim = float(r['similarity'])
            sim_bar = '█' * int(sim * 20) + '░' * (20 - int(sim * 20))
            report.append(f"\n  #{i:>2}  Similarity: {sim:.4f}  [{sim_bar}]")
            report.append(f"      Claude  [{r['claude_mode'] or '?':<12}]: {trunc(r['claude_text'], 90)}")
            report.append(f"      ChatGPT [{r['chatgpt_mode'] or '?':<12}]: {trunc(r['chatgpt_text'], 90)}")

        # Mode pairing analysis
        report.append(subheader("Cross-Platform Mode Pairings"))
        mode_pairs = {}
        for r in semantic_matches:
            pair = f"{r['claude_mode'] or 'unknown'} <-> {r['chatgpt_mode'] or 'unknown'}"
            mode_pairs[pair] = mode_pairs.get(pair, 0) + 1
        for pair, cnt in sorted(mode_pairs.items(), key=lambda x: -x[1]):
            report.append(f"    {pair:<45} {cnt:>4} matches")
    else:
        report.append("  No cross-platform matches found at similarity >= 0.65")

    # ═══════════════════════════════════════════════════════════════════
    # 3. COHERENCE MOMENT SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    report.append(header("3. COHERENCE MOMENTS"))

    coherence_total = await conn.fetchval("SELECT COUNT(*) FROM coherence_moments")
    report.append(f"  Total coherence moments detected: {coherence_total}")

    # By type
    coherence_types = await conn.fetch("""
        SELECT coherence_type, COUNT(*) as cnt,
               AVG(confidence) as avg_conf, MAX(confidence) as max_conf,
               AVG(time_window_s) as avg_window
        FROM coherence_moments
        GROUP BY coherence_type
        ORDER BY cnt DESC
    """)

    report.append(subheader("By Coherence Type"))
    report.append(f"  {'Type':<30} {'Count':>6} {'Avg Conf':>10} {'Max Conf':>10} {'Avg Window':>12}")
    report.append(f"  {'─'*30} {'─'*6} {'─'*10} {'─'*10} {'─'*12}")
    for r in coherence_types:
        report.append(f"  {r['coherence_type']:<30} {r['cnt']:>6} {float(r['avg_conf']):>9.3f} {float(r['max_conf']):>9.3f} {float(r['avg_window']):>10.0f}s")

    # Confidence distribution
    conf_dist = await conn.fetch("""
        SELECT
            CASE
                WHEN confidence >= 0.9 THEN '0.90+ (exceptional)'
                WHEN confidence >= 0.8 THEN '0.80-0.90 (strong)'
                WHEN confidence >= 0.7 THEN '0.70-0.80 (solid)'
                WHEN confidence >= 0.6 THEN '0.60-0.70 (moderate)'
                ELSE '< 0.60 (weak)'
            END as bracket,
            COUNT(*) as cnt
        FROM coherence_moments
        GROUP BY bracket
        ORDER BY bracket DESC
    """)

    report.append(subheader("Confidence Distribution"))
    for r in conf_dist:
        report.append(f"    {r['bracket']:<30} {r['cnt']:>6}  {bar(r['cnt'], coherence_total, 30)}")

    # Platforms involved
    platform_involvement = await conn.fetch("""
        SELECT platforms, COUNT(*) as cnt
        FROM coherence_moments
        GROUP BY platforms
        ORDER BY cnt DESC
    """)

    report.append(subheader("Platform Combinations"))
    for r in platform_involvement:
        plats = r['platforms']
        label = ' + '.join(plats) if plats else 'unknown'
        report.append(f"    {label:<45} {r['cnt']:>6}")

    # Top 10 highest-confidence moments
    top_moments = await conn.fetch("""
        SELECT moment_id, coherence_type, confidence, platforms, description, time_window_s
        FROM coherence_moments
        ORDER BY confidence DESC
        LIMIT 10
    """)

    report.append(subheader("Top 10 Highest-Confidence Moments"))
    for i, r in enumerate(top_moments, 1):
        plats = ' + '.join(r['platforms']) if r['platforms'] else '?'
        report.append(f"\n  #{i:>2}  Confidence: {float(r['confidence']):.3f}  |  Type: {r['coherence_type']}  |  Platforms: {plats}")
        report.append(f"      Window: {r['time_window_s']}s  |  ID: {r['moment_id'][:40]}...")
        if r['description']:
            report.append(f"      Desc: {trunc(r['description'], 100)}")

    # ═══════════════════════════════════════════════════════════════════
    # 4. TOPIC CLUSTERS (Cross-Platform)
    # ═══════════════════════════════════════════════════════════════════
    report.append(header("4. CROSS-PLATFORM TOPIC CLUSTERS"))
    report.append("  Identifying shared themes between platforms via semantic proximity...\n")

    # Use light_layer topics if available, else content-based analysis
    topic_data = await conn.fetch("""
        SELECT ce.platform, ce.light_layer->>'topics' as topics,
               ce.light_layer->>'intent' as intent,
               COUNT(*) as cnt
        FROM cognitive_events ce
        WHERE ce.light_layer IS NOT NULL
          AND ce.light_layer->>'topics' IS NOT NULL
        GROUP BY ce.platform, ce.light_layer->>'topics', ce.light_layer->>'intent'
        ORDER BY cnt DESC
        LIMIT 50
    """)

    if topic_data:
        report.append(subheader("Most Common Topics (from Light Layer)"))
        report.append(f"  {'Platform':<20} {'Topic':<40} {'Intent':<15} {'Count':>6}")
        report.append(f"  {'─'*20} {'─'*40} {'─'*15} {'─'*6}")
        for r in topic_data[:30]:
            report.append(f"  {r['platform']:<20} {trunc(r['topics'] or '', 38):<40} {trunc(r['intent'] or '', 13):<15} {r['cnt']:>6}")
    else:
        report.append("  No light_layer topic data found. Analyzing via content keywords...")

    # Content-based cross-platform keyword analysis
    # Find words/phrases that appear in high-similarity matches
    keyword_query = await conn.fetch("""
        SELECT ce.platform,
               LOWER(SUBSTRING(ec.content_preview FROM 1 FOR 200)) as preview_lower,
               ce.cognitive_mode
        FROM embedding_cache ec
        JOIN cognitive_events ce ON ec.source_event_id = ce.event_id
        WHERE ec.content_preview IS NOT NULL
          AND LENGTH(ec.content_preview) > 20
        ORDER BY RANDOM()
        LIMIT 2000
    """)

    # Simple keyword frequency analysis across platforms
    from collections import Counter
    stop_words = {'the','a','an','is','are','was','were','be','been','being','have','has','had',
                  'do','does','did','will','would','could','should','may','might','shall','can',
                  'to','of','in','for','on','with','at','by','from','as','into','through','during',
                  'before','after','above','below','between','under','again','further','then','once',
                  'here','there','when','where','why','how','all','both','each','few','more','most',
                  'other','some','such','no','nor','not','only','own','same','so','than','too','very',
                  'and','but','or','if','it','its','this','that','these','those','i','me','my','we',
                  'our','you','your','he','she','they','them','their','what','which','who','whom',
                  'about','just','also','like','get','got','one','two','make','want','know','think',
                  'say','see','come','go','take','use','find','give','tell','ask','seem','feel',
                  'try','leave','call','need','keep','let','begin','show','hear','play','run','move',
                  'live','believe','bring','happen','write','provide','sit','stand','lose','pay',
                  'meet','include','continue','set','learn','change','lead','understand','watch',
                  'follow','stop','create','speak','read','allow','add','spend','grow','open',
                  'walk','win','offer','remember','love','consider','appear','buy','wait','serve',
                  'die','send','expect','build','stay','fall','cut','reach','kill','remain','suggest',
                  'raise','pass','sell','require','report','decide','pull','new','up','out','well',
                  'even','way','back','still','work','over','much','using','used','based','right',
                  '','s','t','re','ve','ll','d','m','don','doesn','didn','won','wouldn','couldn',
                  'shouldn','isn','aren','wasn','weren','hasn','haven','hadn'}

    platform_words = {}
    for r in keyword_query:
        plat = r['platform']
        if plat not in platform_words:
            platform_words[plat] = Counter()
        words = [w for w in r['preview_lower'].split() if len(w) > 3 and w.isalpha() and w not in stop_words]
        platform_words[plat].update(words)

    if len(platform_words) >= 2:
        report.append(subheader("Shared High-Frequency Keywords (Cross-Platform)"))

        # Find keywords common to both platforms
        all_platforms = list(platform_words.keys())
        shared_keywords = []

        # Get intersection of top keywords
        for word in set().union(*(set(c.keys()) for c in platform_words.values())):
            counts = {p: platform_words[p].get(word, 0) for p in all_platforms}
            if all(c >= 3 for c in counts.values()):  # appears at least 3x on each platform
                total = sum(counts.values())
                shared_keywords.append((word, total, counts))

        shared_keywords.sort(key=lambda x: -x[1])

        report.append(f"\n  {'Keyword':<25}", )
        plat_headers = ''.join(f'{p:>15}' for p in all_platforms)
        report.append(f"  {'Keyword':<25} {plat_headers} {'Total':>10}")
        report.append(f"  {'─'*25} {'─'*15 * len(all_platforms)} {'─'*10}")
        for word, total, counts in shared_keywords[:30]:
            count_str = ''.join(f'{counts[p]:>15,}' for p in all_platforms)
            report.append(f"  {word:<25} {count_str} {total:>10,}")

        report.append(f"\n  Total shared keywords (>=3 per platform): {len(shared_keywords)}")

        # Platform-unique keywords
        for plat in all_platforms:
            unique = [(w, c) for w, c in platform_words[plat].most_common(200)
                      if all(platform_words[p].get(w, 0) < 2 for p in all_platforms if p != plat)]
            report.append(f"\n  Top keywords unique to [{plat}]:")
            report.append(f"    {', '.join(f'{w}({c})' for w, c in unique[:15])}")

    # ═══════════════════════════════════════════════════════════════════
    # 5. COGNITIVE MODE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    report.append(header("5. COGNITIVE MODE ANALYSIS"))

    # deep_work vs exploration comparison
    mode_comparison = await conn.fetch("""
        SELECT platform, cognitive_mode,
               COUNT(*) as events,
               COUNT(DISTINCT session_id) as sessions,
               AVG(quality_score) as avg_quality,
               AVG(content_length) as avg_content_length
        FROM cognitive_events
        WHERE cognitive_mode IN ('deep_work', 'exploration', 'casual')
        GROUP BY platform, cognitive_mode
        ORDER BY platform, events DESC
    """)

    report.append(subheader("Mode Comparison: deep_work vs exploration vs casual"))
    report.append(f"  {'Platform':<20} {'Mode':<15} {'Events':>8} {'Sessions':>10} {'Avg Quality':>12} {'Avg Length':>12}")
    report.append(f"  {'─'*20} {'─'*15} {'─'*8} {'─'*10} {'─'*12} {'─'*12}")
    for r in mode_comparison:
        avg_q = f"{float(r['avg_quality']):.3f}" if r['avg_quality'] else "N/A"
        avg_l = f"{float(r['avg_content_length']):.0f}" if r['avg_content_length'] else "N/A"
        report.append(f"  {r['platform']:<20} {r['cognitive_mode']:<15} {r['events']:>8,} {r['sessions']:>10,} {avg_q:>12} {avg_l:>12}")

    # Quality score distribution by platform and mode
    quality_dist = await conn.fetch("""
        SELECT platform, cognitive_mode,
               PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY quality_score) as q25,
               PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY quality_score) as q50,
               PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY quality_score) as q75,
               PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY quality_score) as q95
        FROM cognitive_events
        WHERE quality_score IS NOT NULL AND cognitive_mode IS NOT NULL
        GROUP BY platform, cognitive_mode
        ORDER BY platform, cognitive_mode
    """)

    if quality_dist:
        report.append(subheader("Quality Score Percentiles by Platform & Mode"))
        report.append(f"  {'Platform':<20} {'Mode':<15} {'P25':>8} {'P50':>8} {'P75':>8} {'P95':>8}")
        report.append(f"  {'─'*20} {'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for r in quality_dist:
            report.append(f"  {r['platform']:<20} {r['cognitive_mode']:<15} {float(r['q25']):>7.3f} {float(r['q50']):>7.3f} {float(r['q75']):>7.3f} {float(r['q95']):>7.3f}")

    # Content depth analysis (longer content = deeper thinking?)
    depth_analysis = await conn.fetch("""
        SELECT platform, cognitive_mode,
               COUNT(*) FILTER (WHERE content_length > 1000) as deep_events,
               COUNT(*) FILTER (WHERE content_length BETWEEN 200 AND 1000) as medium_events,
               COUNT(*) FILTER (WHERE content_length < 200) as shallow_events,
               COUNT(*) as total
        FROM cognitive_events
        WHERE cognitive_mode IN ('deep_work', 'exploration') AND content_length IS NOT NULL
        GROUP BY platform, cognitive_mode
        ORDER BY platform, cognitive_mode
    """)

    if depth_analysis:
        report.append(subheader("Content Depth Analysis (by length)"))
        report.append(f"  {'Platform':<20} {'Mode':<15} {'Deep(>1K)':>10} {'Medium':>10} {'Shallow(<200)':>14} {'Total':>8}")
        report.append(f"  {'─'*20} {'─'*15} {'─'*10} {'─'*10} {'─'*14} {'─'*8}")
        for r in depth_analysis:
            report.append(f"  {r['platform']:<20} {r['cognitive_mode']:<15} {r['deep_events']:>10,} {r['medium_events']:>10,} {r['shallow_events']:>14,} {r['total']:>8,}")

    # ═══════════════════════════════════════════════════════════════════
    # 6. CROSS-PLATFORM COHERENCE INSIGHTS
    # ═══════════════════════════════════════════════════════════════════
    report.append(header("6. CROSS-PLATFORM COHERENCE INSIGHTS"))

    # Cross-platform matches table
    xplat_matches = await conn.fetch("""
        SELECT COUNT(*) as cnt FROM cross_platform_matches
    """)
    if xplat_matches and xplat_matches[0]['cnt'] > 0:
        report.append(f"  Cross-platform matches in dedicated table: {xplat_matches[0]['cnt']}")

        xplat_detail = await conn.fetch("""
            SELECT * FROM cross_platform_matches
            ORDER BY created_at DESC NULLS LAST
            LIMIT 10
        """)
        if xplat_detail:
            report.append(subheader("Recent Cross-Platform Matches"))
            for r in xplat_detail:
                report.append(f"    {dict(r)}")

    # Coherence links
    coh_links = await conn.fetch("SELECT COUNT(*) as cnt FROM coherence_links")
    if coh_links and coh_links[0]['cnt'] > 0:
        report.append(f"\n  Coherence links: {coh_links[0]['cnt']}")

    # Active coherence
    active_coh = await conn.fetch("SELECT COUNT(*) as cnt FROM active_coherence")
    if active_coh and active_coh[0]['cnt'] > 0:
        report.append(f"  Active coherence entries: {active_coh[0]['cnt']}")

    # Temporal distribution of events
    temporal = await conn.fetch("""
        SELECT platform,
               DATE(created_at) as day,
               COUNT(*) as events
        FROM cognitive_events
        WHERE created_at IS NOT NULL
        GROUP BY platform, DATE(created_at)
        ORDER BY day DESC
        LIMIT 30
    """)

    if temporal:
        report.append(subheader("Recent Daily Activity (Last 15 Days)"))
        # Pivot by date
        from collections import defaultdict
        daily = defaultdict(dict)
        for r in temporal:
            daily[str(r['day'])][r['platform']] = r['events']

        all_plats = sorted(set(r['platform'] for r in temporal))
        plat_header = ''.join(f'{p:>18}' for p in all_plats)
        report.append(f"  {'Date':<12} {plat_header} {'Total':>10}")
        report.append(f"  {'─'*12} {'─'*18 * len(all_plats)} {'─'*10}")
        for day in sorted(daily.keys(), reverse=True)[:15]:
            counts = daily[day]
            count_str = ''.join(f'{counts.get(p, 0):>18,}' for p in all_plats)
            total = sum(counts.values())
            report.append(f"  {day:<12} {count_str} {total:>10,}")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    report.append(header("EXECUTIVE SUMMARY"))

    report.append(f"""
  DATA SCALE
    Total cognitive events:     {total_events:>10,}
    Total sessions:             {total_sessions:>10,}
    Total embeddings:           {total_embedded:>10,}
    Coherence moments detected: {coherence_total:>10,}
    Cross-platform matches:     {len(semantic_matches) if semantic_matches else 0:>10,} (at >= 0.65 similarity)

  KEY FINDINGS""")

    if semantic_matches:
        max_sim = max(float(r['similarity']) for r in semantic_matches)
        avg_sim = sum(float(r['similarity']) for r in semantic_matches) / len(semantic_matches)
        report.append(f"    - Strongest cross-platform similarity: {max_sim:.4f}")
        report.append(f"    - Average cross-platform similarity:   {avg_sim:.4f}")

    if coherence_types:
        dominant_type = coherence_types[0]
        report.append(f"    - Dominant coherence type: {dominant_type['coherence_type']} ({dominant_type['cnt']} moments)")

    # Platforms with most shared context
    if semantic_matches:
        high_sim_count = sum(1 for r in semantic_matches if float(r['similarity']) >= 0.80)
        report.append(f"    - High-confidence bridges (>= 0.80): {high_sim_count}")

    report.append(f"""
  COHERENCE HEALTH
    Embedding coverage:  {pct(total_embedded, total_events)} of events have embeddings
    Cross-platform link: {'STRONG' if semantic_matches and max_sim >= 0.85 else 'MODERATE' if semantic_matches and max_sim >= 0.75 else 'DEVELOPING'}
    Data readiness:      {'PRODUCTION' if total_events > 10000 else 'GROWING'}
""")

    report.append(f"{'━' * 80}")
    report.append(f"  Report generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report.append(f"  Database: ucw_cognitive | Platforms: {', '.join(p['platform'] for p in platform_events)}")
    report.append(f"{'━' * 80}")

    await conn.close()

    full_report = '\n'.join(report)
    print(full_report)

    # Save report to file
    report_path = '/Users/dicoangelo/researchgravity/coherence_report.txt'
    with open(report_path, 'w') as f:
        f.write(full_report)
    print(f"\nReport saved to: {report_path}")

if __name__ == '__main__':
    asyncio.run(run_analysis())
