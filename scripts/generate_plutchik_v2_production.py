import csv, json, random
import pandas as pd
from transformers import AutoTokenizer
from collections import defaultdict, Counter
random.seed(2024)

PLUTCHIK = {
    "joy":{"ring":"primary","sector":"joy","color":"F4D03F"},
    "trust":{"ring":"primary","sector":"trust","color":"27AE60"},
    "fear":{"ring":"primary","sector":"fear","color":"196F3D"},
    "surprise":{"ring":"primary","sector":"surprise","color":"2E86C1"},
    "sadness":{"ring":"primary","sector":"sadness","color":"2980B9"},
    "disgust":{"ring":"primary","sector":"disgust","color":"AF7AC5"},
    "anger":{"ring":"primary","sector":"anger","color":"E74C3C"},
    "anticipation":{"ring":"primary","sector":"anticipation","color":"E67E22"},
    "ecstasy":{"ring":"intense","sector":"joy","color":"F1C40F"},
    "admiration":{"ring":"intense","sector":"trust","color":"1E8449"},
    "terror":{"ring":"intense","sector":"fear","color":"145A32"},
    "amazement":{"ring":"intense","sector":"surprise","color":"1A5276"},
    "grief":{"ring":"intense","sector":"sadness","color":"1B2631"},
    "loathing":{"ring":"intense","sector":"disgust","color":"6C3483"},
    "rage":{"ring":"intense","sector":"anger","color":"C0392B"},
    "vigilance":{"ring":"intense","sector":"anticipation","color":"CA6F1E"},
    "serenity":{"ring":"mild","sector":"joy","color":"F9E79F"},
    "acceptance":{"ring":"mild","sector":"trust","color":"ABEBC6"},
    "apprehension":{"ring":"mild","sector":"fear","color":"A9DFBF"},
    "distraction":{"ring":"mild","sector":"surprise","color":"AED6F1"},
    "pensiveness":{"ring":"mild","sector":"sadness","color":"D6EAF8"},
    "boredom":{"ring":"mild","sector":"disgust","color":"E8DAEF"},
    "annoyance":{"ring":"mild","sector":"anger","color":"F5CBA7"},
    "interest":{"ring":"mild","sector":"anticipation","color":"FDEBD0"},
    "optimism":{"ring":"dyadic","sector":"joy+anticipation","color":"F0B27A"},
    "love":{"ring":"dyadic","sector":"joy+trust","color":"82E0AA"},
    "submission":{"ring":"dyadic","sector":"trust+fear","color":"76D7C4"},
    "awe":{"ring":"dyadic","sector":"fear+surprise","color":"7FB3D3"},
    "disapproval":{"ring":"dyadic","sector":"surprise+sadness","color":"7D9EC0"},
    "remorse":{"ring":"dyadic","sector":"sadness+disgust","color":"C39BD3"},
    "contempt":{"ring":"dyadic","sector":"disgust+anger","color":"E59866"},
    "aggressiveness":{"ring":"dyadic","sector":"anger+anticipation","color":"F1948A"},
}

IAA = {
    "ecstasy":0.95,"grief":0.95,"rage":0.93,"terror":0.93,"loathing":0.92,
    "admiration":0.92,"joy":0.90,"sadness":0.90,"anger":0.88,"fear":0.88,
    "disgust":0.87,"surprise":0.85,"trust":0.85,"amazement":0.85,"vigilance":0.82,
    "anticipation":0.83,"remorse":0.82,"love":0.83,"acceptance":0.80,"optimism":0.80,
    "apprehension":0.78,"serenity":0.78,"annoyance":0.77,"pensiveness":0.75,
    "contempt":0.75,"boredom":0.74,"aggressiveness":0.73,"interest":0.72,
    "disapproval":0.72,"awe":0.70,"distraction":0.68,"submission":0.65,
}

POLARITY = {
    "ecstasy":"positive","joy":"positive","serenity":"positive","admiration":"positive",
    "trust":"positive","acceptance":"positive","love":"positive","optimism":"positive",
    "amazement":"positive","interest":"positive","anticipation":"positive",
    "vigilance":"positive","awe":"positive","submission":"neutral","surprise":"neutral",
    "distraction":"neutral","grief":"negative","rage":"negative","terror":"negative",
    "loathing":"negative","sadness":"negative","anger":"negative","fear":"negative",
    "disgust":"negative","remorse":"negative","contempt":"negative","aggressiveness":"negative",
    "disapproval":"negative","apprehension":"negative","pensiveness":"negative",
    "annoyance":"negative","boredom":"negative",
}

EMOJI_MAP = {
    "joy": ["😊", "😄", "✨", "🙌"],
    "trust": ["🤝", "✅", "🛡️", "🙏"],
    "fear": ["😨", "😰", "⚠️", "🧤"],
    "surprise": ["😲", "‼️", "⚡", "🧩"],
    "sadness": ["😢", "😞", "💧", "📉"],
    "disgust": ["🤢", "🤮", "👎", "🍄"],
    "anger": ["😠", "💢", "🔥", "😤"],
    "anticipation": ["⏳", "🔍", "📅", "🔭"],
    "ecstasy": ["🤩", "🥳", "🌈", "🔥"],
    "admiration": ["👏", "🌟", "🎖️", "🫡"],
    "terror": ["😱", "💀", "🆘", "🧟"],
    "amazement": ["🤯", "🌀", "🌟", "🎇"],
    "grief": ["😭", "🥀", "🕯️", "💔"],
    "loathing": ["🤬", "👺", "🤮", "💀"],
    "rage": ["😡", "💥", "🌋", "🧨"],
    "vigilance": ["🕵️", "🧐", "🚨", "🛡️"],
    "serenity": ["😌", "🧘", "🍃", "🌊"],
    "acceptance": ["👌", "🆗", "👍", "🤝"],
    "apprehension": ["😰", "😟", "🩹", "🌪️"],
    "distraction": ["❓", "😵‍💫", "🌀", "🛰️"],
    "pensiveness": ["🤔", "💭", "📓", "⌛"],
    "boredom": ["🥱", "😑", "💤", "🕸️"],
    "annoyance": ["😒", "🙄", "💢", "💨"],
    "interest": ["💡", "🧐", "🎯", "🔭"],
    "optimism": ["🚀", "🌅", "📈", "☀️"],
    "love": ["❤️", "💖", "🥰", "💞"],
    "submission": ["🙇", "🏳️", "🤐", "🤝"],
    "awe": ["🙌", "🌠", "🌌", "🏙️"],
    "disapproval": ["🚫", "🙅", "👎", "❌"],
    "remorse": ["😔", "🙏", "💔", "🥀"],
    "contempt": ["😏", "💅", "🙄", "🤨"],
    "aggressiveness": ["⚔️", "🧨", "😤", "🥊"],
}

# (speaker, text, emotion, sarcasm_flag, emotion_cause)
DIALOGUES = [
# ══════════════ WORKPLACE ══════════════
{"scenario":"workplace","topic":"termination","utterances":[
("Manager","I need to inform you that your position is being eliminated effective this Friday.","sadness",False,"sudden job elimination announced"),
("Employee","I have been with this organisation for six years without a single negative performance review.","rage",False,"loyal service met with abrupt termination"),
("Manager","This decision came directly from the executive team and was not about your performance at all.","acceptance",False,None),
("Employee","Six years of my life and you give me three days notice — that is quite the send-off.","contempt",True,"disrespectfully short notice given"),
("Manager","We are offering a generous severance package and full outplacement support from today.","trust",False,"severance offered to partially compensate"),
("Employee","A severance package does not replace six years of career investment in this company.","grief",False,"career investment feels permanently wasted"),
("Manager","I understand your anger and I genuinely believe this decision is unjust as well.","sadness",False,"manager privately shares employee's view"),
("Employee","Then why are you the one sitting across this table delivering the news to me.","anger",False,"manager complicit in unjust decision"),
]},
{"scenario":"workplace","topic":"bonus","utterances":[
("Director","The board approved a twenty percent bonus for the entire engineering organisation.","ecstasy",False,"substantial bonus approved by board"),
("Senior_Dev","I genuinely did not see that coming after the difficulty of this past quarter.","amazement",False,"unexpected reward after gruelling period"),
("Director","Your team shipped three products under budget and ahead of every scheduled milestone.","admiration",False,"exceptional delivery recognised publicly"),
("Junior_Dev","Does that include contractors or only full-time employees in this cycle.","interest",False,"seeking clarity on bonus eligibility"),
("Director","Full-time only this cycle but we are actively advocating for contractor inclusion next year.","trust",False,"honest disclosure about contractor exclusion"),
("Senior_Dev","This kind of recognition is exactly what makes people genuinely want to stay here.","joy",False,"recognition reinforcing retention decision"),
("Junior_Dev","I have been underpaid for two years but I will take this as a meaningful starting point.","optimism",False,"bonus signals better future ahead"),
]},
{"scenario":"workplace","topic":"harassment","utterances":[
("HR_Manager","Several colleagues have independently reported identical patterns of behaviour from the director.","vigilance",False,"multiple corroborating harassment reports received"),
("Employee_A","I reported this exact issue three months ago and was told to work it out directly with him.","rage",False,"previous report dismissed without action"),
("HR_Manager","That response was completely wrong and I apologise unreservedly on behalf of this organisation.","remorse",False,"HR acknowledges prior institutional failure"),
("Employee_B","What actually happens now because that man still walks into this building every single morning.","fear",False,"harasser continuing normal access to workplace"),
("HR_Manager","He is suspended pending a full independent investigation that begins today without exception.","trust",False,"immediate protective action taken"),
("Employee_A","I have watched other women leave this company because of him and I said nothing each time.","grief",False,"guilt over silence while others suffered and left"),
("HR_Manager","Your speaking up today has protected people who do not yet know they were at risk.","admiration",False,"courage in reporting acknowledged directly"),
]},
{"scenario":"workplace","topic":"underpayment","utterances":[
("Employee_A","I discovered that every male peer in my salary band earns fifteen percent more than I do.","rage",False,"gender pay gap personally discovered"),
("Employee_B","That is not a rounding error or an oversight — that is a deliberate structural policy.","contempt",False,"pay gap identified as intentional practice"),
("Employee_A","HR told me pay is based on negotiation history and individual performance calibration scores.","disgust",False,"HR explanation felt like deliberate misdirection"),
("Employee_B","Translation — they rewarded whoever negotiated hardest and called that meritocracy.","contempt",True,"satirising hollow meritocracy framing"),
("Employee_A","I have the highest performance rating in this team for three consecutive annual cycles.","anger",False,"strong performance not reflected in compensation"),
("Employee_B","Document every data point and get an employment lawyer to review before you escalate formally.","vigilance",False,"strategic advice to build strongest possible case"),
("Employee_A","I am not leaving quietly and I am not going to pretend this situation is remotely acceptable.","aggressiveness",False,"commitment to fight pay discrimination openly"),
]},
{"scenario":"workplace","topic":"whistleblowing","utterances":[
("Analyst","The financial statements being submitted to regulators contain fabricated revenue figures for three quarters.","vigilance",False,"serious accounting fraud personally discovered"),
("Compliance_Lead","You need to be completely certain before you take this any further with anyone.","apprehension",False,"severity of accusation fully recognised"),
("Analyst","I have three years of reconciliation files that show precisely where the numbers diverge from reality.","trust",False,"evidence carefully compiled over time"),
("Compliance_Lead","If you are right then every executive who signed those filings is personally and criminally liable.","amazement",False,"full scale of fraud becoming clear"),
("Analyst","I know exactly what reporting this means for my career and I am reporting it anyway.","vigilance",False,"personal professional risk consciously accepted"),
("Compliance_Lead","I will go with you to the external regulator — you should not walk into that office alone.","love",False,"personal support offered in dangerous act"),
("Analyst","I have been losing sleep over this for two months and I simply cannot carry it any longer.","grief",False,"prolonged moral burden from sustained silence"),
]},
{"scenario":"workplace","topic":"burnout","utterances":[
("Employee","I submitted a two-week medical leave request for burnout and exhaustion this morning.","sadness",False,"burnout serious enough to require medical leave"),
("Manager","I am glad you did that and I am genuinely sorry you reached this point on my watch.","remorse",False,"manager feels responsible for employee's burnout"),
("Employee","I tried to signal the problem twice but the signals I sent were far too subtle to land.","remorse",False,"regret over unclear early communication"),
("Manager","I should have asked you directly and explicitly how you were holding up — that is entirely on me.","remorse",False,"manager accepts failure to monitor wellbeing"),
("Employee","I am not angry at you — I am just completely empty and I need time to recover properly.","sadness",False,"emotional exhaustion beyond anger"),
("Manager","Take the full two weeks and do not open your email once — I mean that as a direct instruction.","trust",False,"explicit permission to fully disconnect given"),
("Employee","Can we talk about workload structure when I return because something fundamental has to change here.","anticipation",False,"planning structural reform after recovery period"),
]},
{"scenario":"workplace","topic":"micromanagement","utterances":[
("Developer","She has added herself to every single calendar invite I have for the past six straight weeks.","annoyance",False,"pervasive and escalating micromanagement"),
("Colleague","Does she review your pull requests before your own team lead does now as well.","interest",True,"sarcastic question probing micromanagement depth"),
("Developer","She left eighteen comments on a prototype that was never scheduled to ship to anyone.","disgust",False,"micromanagement reaching absurd level"),
("Colleague","Eighteen comments on a throwaway prototype is genuinely extraordinary in the worst way.","amazement",False,"disbelief at depth of micromanagement"),
("Developer","I cannot concentrate because every single decision feels like a test I might fail publicly.","fear",False,"surveillance creating constant performance anxiety"),
("Colleague","Have you tried naming the dynamic directly without naming the specific behaviour pattern.","interest",False,"suggesting diplomatic framing for conversation"),
("Developer","I tried once and she described herself as being thorough and supportive in her approach.","contempt",False,"feedback dismissed with hollow self-description"),
("Colleague","Then you document the pattern carefully and escalate it formally or you start looking outward.","vigilance",False,"advising escalation or strategic departure"),
]},
{"scenario":"workplace","topic":"innovation","utterances":[
("Founder","Every single investor we pitched to last week passed on us without serious engagement.","sadness",False,"unanimous investor rejection received"),
("CTO","All five of them said the market is too small — every one of them used that exact phrase.","grief",False,"repeated market size objection feels definitive"),
("Founder","They cannot see what we are building and they have already decided based on category alone.","contempt",False,"surface-level judgment before seeing product"),
("CTO","Maybe we are pitching to entirely the wrong category of investor for this particular thesis.","interest",False,"questioning fundamental investor fit"),
("Founder","That is actually a very good point — we have been targeting pure SaaS growth funds exclusively.","amazement",False,"realisation about fundamentally wrong investor type"),
("CTO","Deep tech and climate infrastructure funds might read our thesis in a completely different way.","anticipation",False,"more appropriate investor category identified"),
("Founder","Let me rebuild the entire target list by Friday and we go again with completely fresh energy.","optimism",False,"commitment to pivot pitch strategy entirely"),
]},
{"scenario":"workplace","topic":"nepotism","utterances":[
("Employee_A","The CEO hired his nephew as Head of Product over four strong internal candidates this week.","disgust",False,"nepotism observed at senior leadership level"),
("Employee_B","The nephew has eighteen months of product experience — a truly inspired appointment.","contempt",True,"sarcasm highlighting absurdly underqualified hire"),
("Employee_A","One of those internal candidates has been here eight years and shipped twelve successful products.","anger",False,"highly deserving candidate unjustly bypassed"),
("Employee_B","I have stopped being surprised by this and started being strategic about my own future options.","acceptance",False,"pragmatic adaptation to systemic unfairness"),
("Employee_A","The problem is that being strategic means leaving a company I have genuinely loved working for.","sadness",False,"forced choice between integrity and attachment"),
("Employee_B","Sometimes love for a place and clear-eyed understanding of its ceiling are not mutually exclusive feelings.","trust",False,"wisdom about holding both truths simultaneously"),
("Employee_A","I am going to update my CV this weekend — not as a dramatic statement but simply as a hedge.","vigilance",False,"cautious preparation without emotional escalation"),
]},
{"scenario":"workplace","topic":"crisis","utterances":[
("Incident_Lead","Production is completely down and sixty thousand active users cannot access their data right now.","terror",False,"large-scale production outage in progress"),
("Senior_Eng","The primary database cluster failed over but the replica was three hours behind in sync.","vigilance",False,"root cause of outage precisely identified"),
("Incident_Lead","How long to restore service from the last verified clean backup we have available.","interest",False,"assessing realistic recovery timeline"),
("Senior_Eng","Four hours minimum — the backup validation process is the bottleneck not the restore itself.","sadness",False,"long recovery time confirmed with specifics"),
("Incident_Lead","Draft customer communication now and assume the four-hour window in your messaging today.","vigilance",False,"proactive customer communication ordered immediately"),
("Senior_Eng","The reason this replica was stale is because someone disabled the sync job last Tuesday.","rage",False,"preventable failure caused by undocumented change"),
("Incident_Lead","We investigate cause after we restore full service — in exactly that order every single time.","trust",False,"process discipline maintained under severe pressure"),
]},
{"scenario":"workplace","topic":"demotion","utterances":[
("Senior_Mgr","We are restructuring your role into a senior individual contributor position going forward.","surprise",False,"unexpected role restructure announced"),
("Team_Lead","That is a demotion by every meaningful definition except the specific word you are choosing.","anger",False,"euphemistic framing of demotion challenged"),
("Senior_Mgr","The new structure reflects where the company most needs depth and expertise right now.","trust",False,"restructure framed as strategic alignment"),
("Team_Lead","I turned down two external offers in the past year because I trusted the path here explicitly.","grief",False,"loyalty rewarded with effective demotion"),
("Senior_Mgr","Those decisions were yours and I am not dismissing how deeply unfair this moment feels.","acceptance",False,None),
("Team_Lead","I need twenty-four hours before I can have a professional conversation about the details.","vigilance",False,"needs processing time before responding"),
("Senior_Mgr","That is completely fair and reasonable — take the time you need without pressure.","trust",False,None),
]},
{"scenario":"workplace","topic":"acquisition","utterances":[
("CEO","The acquisition was announced this morning and our entire company is now owned by our main competitor.","amazement",False,"company sold to direct competitor"),
("Lead_Eng","We built this product for five years specifically to compete against them — this is genuinely surreal.","distraction",False,"disorientation from completely unexpected outcome"),
("CEO","The terms are very favourable and every employee receives full vesting acceleration effective today.","joy",False,"employee-friendly acquisition terms announced"),
("Lead_Eng","I do not know whether to feel relieved that it is over or betrayed that this is how it ends.","distraction",False,"conflicting emotions about unexpected outcome"),
("CEO","The acquiring team specifically requested that the entire engineering organisation stay completely intact.","trust",False,"team explicitly valued by acquiring company"),
("Lead_Eng","I need at least a week to process this fully before I can decide what I genuinely want to do.","acceptance",False,None),
("CEO","That is entirely reasonable and your position is completely secure while you think it through.","serenity",False,None),
]},
{"scenario":"workplace","topic":"resignation","utterances":[
("Manager","I was not expecting your resignation letter to arrive today — I will be honest with you about that.","surprise",False,"unexpected resignation received"),
("Employee","I have been unhappy here for a long time and I waited far too long to say something direct.","sadness",False,"delayed recognition of long-standing dissatisfaction"),
("Manager","What would have changed this outcome if I had actually known about it sooner.","interest",False,"seeking honest feedback to understand"),
("Employee","More genuine autonomy and fewer status meetings that replaced actual productive work time.","trust",False,"specific and honest feedback given"),
("Manager","I hear that and I genuinely wish we had had this exact conversation nine months ago.","remorse",False,"regret over missed opportunity to retain"),
("Employee","I am leaving on genuinely good terms — this company gave me a great deal and I mean that.","acceptance",False,None),
("Manager","Where are you headed if you are comfortable sharing that with me.","interest",False,None),
("Employee","Somewhere smaller where I can actually see the direct impact of everything I build every day.","optimism",False,"seeking more meaningful and visible impact"),
]},
{"scenario":"workplace","topic":"plagiarism_work","utterances":[
("Manager","Your client presentation and Marcus's pitch deck are seventy percent structurally identical.","vigilance",False,"strong evidence of plagiarism discovered"),
("Employee","He sat directly next to me for three hours while I built that entire deck last Tuesday.","rage",False,"plagiarist was physically present during creation"),
("Manager","Why did you not flag it the moment you saw his version circulating in the organisation.","interest",False,"questioning delay in raising concern"),
("Employee","Because I thought I was losing my mind and nobody would believe me over someone his level.","fear",False,"power imbalance made reporting feel genuinely unsafe"),
("Manager","I believe you right now and so will everyone who examines the file metadata timestamps.","trust",False,"evidence strongly supports employee's account"),
("Employee","That presentation has my client relationship embedded on every single slide and he walked in alone.","loathing",False,"deep betrayal by trusted senior colleague"),
("Manager","This is being escalated to HR and legal today — do not contact him directly under any circumstances.","vigilance",False,"formal escalation initiated immediately"),
]},
{"scenario":"workplace","topic":"disability","utterances":[
("Employee","I requested a standing desk adjustment six weeks ago and have received nothing but promises.","annoyance",False,"reasonable adjustment request ignored"),
("HR","The procurement process for specialist equipment requires three separate levels of approval.","trust",False,"bureaucratic process cited as explanation"),
("Employee","My condition causes significant pain after forty minutes of sitting and I am in pain every afternoon.","sadness",False,"daily physical suffering due to inaction"),
("HR","I was not aware the health impact was this immediate and significant — I apologise for that.","remorse",False,"impact not communicated upward"),
("Employee","I included that information in the initial request form and in both follow-up emails I sent.","anger",False,"documented communication had been ignored"),
("HR","I am escalating this as a health and safety matter today rather than a procurement request.","vigilance",False,"reclassifying to expedite resolution"),
("Employee","Six weeks of pain because of a category on a form — I need you to understand how that feels.","grief",False,"systemic indifference producing real suffering"),
]},

# ══════════════ FRIENDSHIP ══════════════
{"scenario":"friendship","topic":"confession","utterances":[
("Friend_1","I need to tell you something I have been sitting with for three months and I cannot any longer.","apprehension",False,"secret carried beyond bearable point"),
("Friend_2","You are scaring me a little right now — please just say what it is.","fear",False,"serious tone producing anticipatory anxiety"),
("Friend_1","I was the person who told your ex where you were living after you moved last year.","loathing",True,"admission of devastating betrayal"),
("Friend_2","You told him where I was after I spent six months hiding from him for my own safety.","rage",False,"safety-threatening betrayal fully revealed"),
("Friend_1","I did not know the full picture at all — I thought you had ended things normally.","remorse",False,"action taken without critical safety context"),
("Friend_2","I told you directly that it was not a safe situation and you still gave him my location.","anger",False,"warning was given clearly and ignored"),
("Friend_1","I have lived with this guilt every single day since I found out what he had actually been doing.","remorse",False,"profound guilt after learning true consequences"),
("Friend_2","I need you to leave right now and I need significant time before I can look at you again.","grief",False,"friendship may be irreparably destroyed"),
]},
{"scenario":"friendship","topic":"intervention","utterances":[
("Friend_1","We are all here together because we are genuinely terrified for you and we love you enormously.","love",False,"group intervention driven purely by care"),
("Subject","I knew something was happening when you all suggested lunch at the same time like that.","apprehension",False,"group gathering felt staged and worrying"),
("Friend_2","You have lost twelve kilograms in eight weeks and you honestly think we would not notice that.","sadness",False,"visible physical deterioration observed"),
("Subject","I am fine — I have just been working far too hard and not prioritising eating enough.","distraction",False,"minimising behaviour to deflect concern"),
("Friend_1","We are not doing this to embarrass you — we are doing this because we are genuinely terrified.","terror",False,"deep fear for friend's life driving action"),
("Subject","If I actually admit how bad it has gotten then I have to deal with it and I am not ready.","fear",False,"avoidance driven by fear of consequences"),
("Friend_2","You do not have to be ready alone — that is literally why every one of us is sitting here.","love",False,"collective unconditional support offered"),
("Subject","I know that I need help. I have known for quite a while now.","acceptance",False,"first honest acknowledgement of need"),
]},
{"scenario":"friendship","topic":"jealousy","utterances":[
("Friend_1","I will be completely honest with you — I felt a flash of jealousy when you announced the book deal.","remorse",False,"jealousy honestly admitted without deflection"),
("Friend_2","I appreciate you saying that directly rather than just smiling warmly and hiding it.","admiration",False,"honesty in friendship genuinely valued"),
("Friend_1","You have worked toward this specific goal for ten years and my reaction should have been pure joy.","remorse",False,"shame over mixed jealous reaction"),
("Friend_2","Jealousy does not make you a bad friend — it means you want something meaningful for yourself too.","trust",False,"complex emotion normalised without judgment"),
("Friend_1","I think it surfaced all the ways I have been deferring my own creative work indefinitely.","interest",False,"jealousy as signal about unmet personal aspiration"),
("Friend_2","Then use it as information — let it tell you something useful instead of making you feel guilty.","optimism",False,"reframing jealousy as actionable signal"),
("Friend_1","I am genuinely so proud of you and I am sorry the two feelings got tangled up together.","love",False,"separating pride from jealousy successfully"),
]},
{"scenario":"friendship","topic":"growing_apart","utterances":[
("Friend_1","I think we have been performing a closeness that we are not actually maintaining anymore.","pensiveness",False,"honest recognition of relationship drift"),
("Friend_2","That is a painful thing to say out loud but you are absolutely not wrong about it.","sadness",False,"truth acknowledged despite its difficulty"),
("Friend_1","Our lives have just gone in genuinely different directions in these last three years.","acceptance",False,None),
("Friend_2","The caring is entirely still there — that part has not changed at all for me.","love",False,None),
("Friend_1","The caring is there but we have stopped actually knowing who the other person is becoming.","sadness",False,"relationship existing primarily in past tense"),
("Friend_2","Is this a goodbye conversation or is it a wake-up call conversation — I need to understand.","interest",False,"seeking to understand the intent here"),
("Friend_1","It is a conversation where we are honest about where we are and see what we can rebuild together.","trust",False,"honesty as foundation for possible renewal"),
("Friend_2","Then I am fully in for that conversation — every honest bit of it.","optimism",False,"genuine willingness to rebuild honestly"),
]},
{"scenario":"friendship","topic":"awe_nature","utterances":[
("Hiker_1","I have been to sixty countries and I have never seen anything in my life like what we are looking at.","awe",False,"natural landscape exceeding all prior experience"),
("Hiker_2","We walked for nine hours today and I would repeat every single step for this exact moment.","ecstasy",False,"extreme effort entirely justified by experience"),
("Hiker_1","Standing in front of this makes every problem I brought with me feel genuinely and completely tiny.","awe",False,"natural scale producing profound perspective shift"),
("Hiker_2","The scale of it is almost frightening — do you feel that particular quality in it too.","awe",False,"sublimity producing fear-wonder blend"),
("Hiker_1","Something this large has no business being this beautiful — those two things should not coexist.","amazement",False,"scale and beauty experienced as impossible combination"),
("Hiker_2","I cannot stop looking and simultaneously cannot fully take it in — both feelings at once.","awe",False,"incomprehensible beauty overwhelming perceptual capacity"),
("Hiker_1","This is why we do this — to feel completely alive and completely small at the exact same moment.","serenity",False,"meaning found in natural transcendent experience"),
]},
{"scenario":"friendship","topic":"reconciliation","utterances":[
("Friend_1","Two years of complete silence and you are the one who finally reached out — why specifically now.","interest",False,"questioning motive for reconnection after long gap"),
("Friend_2","Because I had a health scare and realised I had let something irreplaceable quietly decay.","sadness",False,"mortality prompting reconnection with what matters"),
("Friend_1","I thought about reaching out so many times and always found a reason not to take the step.","remorse",False,"repeated inaction despite genuine desire to reconnect"),
("Friend_2","The argument that ended us was not worth two years of this silence — it simply was not.","acceptance",False,"falling out put in appropriate perspective"),
("Friend_1","I was so convinced I was right that I did not notice how lonely I was steadily becoming.","sadness",False,"pride as mechanism of self-imposed isolation"),
("Friend_2","I am not remotely interested in who was right — I am interested in whether we still have something.","trust",False,"moving past blame toward genuine possibility"),
("Friend_1","We do — I knew that the moment I read your message and my chest loosened.","love",False,"friendship recognised as still alive and worth saving"),
]},
{"scenario":"friendship","topic":"awe_achievement","utterances":[
("Observer_1","She solved a problem that four dedicated research teams failed to crack across an entire decade.","awe",False,"extraordinary intellectual achievement witnessed"),
("Observer_2","I sat three rows back when she presented and I genuinely could not breathe properly.","amazement",False,"physical response to witnessing landmark work"),
("Observer_1","The entire room went completely silent for about thirty seconds after she finished the proof.","awe",False,"collective recognition of monumental scientific moment"),
("Observer_2","There are moments in a career when you know you are in the presence of something genuinely historic.","awe",False,"historical significance of moment fully recognised"),
("Observer_1","She just stood there waiting for questions as if she had done something entirely ordinary.","admiration",False,"contrast between achievement and profound humility"),
("Observer_2","I feel genuinely lucky that I happened to be in that specific room when it occurred.","awe",False,"gratitude for witnessing rare pivotal event"),
]},
{"scenario":"friendship","topic":"submission_conflict","utterances":[
("Friend_1","I have decided to drop my legal case against the landlord even though I am completely in the right.","submission",False,"choosing peace and sanity over vindication"),
("Friend_2","You are completely certain the documentation supports your position thoroughly.","trust",False,"verifying that surrender is strategic not mistaken"),
("Friend_1","Entirely certain — but the legal process will cost me two years of my life and my mental health.","acceptance",False,"strategic surrender chosen with clear eyes"),
("Friend_2","So you are stepping down not because you are wrong but because the fight is no longer worth it.","interest",False,"distinguishing yielding from admitting fault"),
("Friend_1","Precisely. I am choosing my wellbeing and my future over the satisfaction of winning publicly.","submission",False,"conscious deliberate choice to defer to reality"),
("Friend_2","That takes more genuine strength than simply fighting it out would have.","admiration",False,"wisdom in surrender recognised"),
("Friend_1","I am still angry about the injustice and I am simultaneously at peace with the decision.","acceptance",False,"anger and peace coexisting without contradiction"),
]},
{"scenario":"friendship","topic":"disapproval_choices","utterances":[
("Friend_1","I am going to be honest with you because I genuinely care about what happens to you.","disapproval",False,"moral disapproval expressed from place of care"),
("Friend_2","You are about to talk about my relationship which is my business and not yours.","anger",False,"boundary asserted before judgment lands"),
("Friend_1","You are involved with someone who is married with children and I cannot pretend that is acceptable.","disapproval",False,"moral judgment stated plainly and directly"),
("Friend_2","I know exactly how it looks and I am not proud of it but I am in it deeply now.","remorse",False,"shame acknowledged without ending the situation"),
("Friend_1","The children in that family are going to be collateral damage in this situation regardless.","disapproval",False,"concern extended to uninvolved parties who will be harmed"),
("Friend_2","That specific thought haunts me every single day — you do not need to tell me about it.","grief",False,"moral weight of situation carried alone daily"),
("Friend_1","I am not going anywhere as your friend — I just needed you to know I see this clearly.","love",False,"honesty offered without abandonment"),
]},
{"scenario":"friendship","topic":"sarcasm_social","utterances":[
("Friend_1","Oh brilliant — another dinner party where we all perform enthusiasm about the same wine.","boredom",True,"sarcasm about repetitive hollow social ritual"),
("Friend_2","And spend forty-five minutes discussing someone's kitchen renovation in extraordinary detail.","boredom",True,"mocking perfectly predictable party conversation"),
("Friend_1","At least Marcus will mention his half-marathon again so we have that to look forward to.","contempt",True,"mockery of friend's endlessly repeated boasting"),
("Friend_2","I genuinely cannot remember a social occasion where Marcus did not mention his half-marathon.","boredom",False,"repetition so complete it has become universal law"),
("Friend_1","Maybe we arrive at eight and claim an early morning commitment by nine-thirty at the latest.","anticipation",False,"planning minimal attendance strategy together"),
("Friend_2","Perfect. I will send you a fake emergency text at nine-fifteen just in case it runs long.","joy",False,"conspiring cheerfully for dignified early exit"),
("Friend_1","This is precisely why you are my absolute favourite person in the world.","love",False,"deep appreciation for perfect social compatibility"),
]},
{"scenario":"friendship","topic":"bereavement","utterances":[
("Friend_1","I did not know what to say when your father died so I said nothing and I am deeply ashamed.","remorse",False,"guilt over silence during friend's worst grief"),
("Friend_2","I noticed you went quiet and it hurt in a way I had not anticipated at all.","sadness",False,"absence painfully felt during profound grief"),
("Friend_1","I was so frightened of saying the wrong thing that I chose to say nothing at all instead.","apprehension",False,"fear of inadequacy produced complete avoidance"),
("Friend_2","The wrong thing would have been infinitely better than the silence that followed.","anger",False,"absence more damaging than imperfect support"),
("Friend_1","I understand that now completely. I am truly sorry without any qualification.","remorse",False,"sincere unqualified apology for failure"),
("Friend_2","I needed to say this because I want to keep this friendship and honesty was the only way.","trust",False,"honesty chosen specifically to preserve friendship"),
("Friend_1","Thank you for giving me that chance. I will not disappear from you again.","love",False,"commitment to reliable presence in future"),
]},
{"scenario":"friendship","topic":"awe_art_friend","utterances":[
("Visitor_1","I have seen reproductions of this painting my entire life and the original is nothing like them.","awe",False,"first encounter with original artwork vs reproduction"),
("Visitor_2","The scale alone is the first thing — no reproduction communicates the actual physical size.","amazement",False,"physical scale of masterwork producing wonder"),
("Visitor_1","There is something breathing in the actual brushwork that I cannot explain to myself logically.","awe",False,"physical proximity revealing intangible aliveness"),
("Visitor_2","I have been in this room for forty minutes and I keep looking up from my phone involuntarily.","amazement",False,"art repeatedly arresting habitual phone use"),
("Visitor_1","I think this is what they mean when people say great art is alive — it is actively doing something.","awe",False,"understanding art's aliveness through direct experience"),
("Visitor_2","I want to come back tomorrow morning when the room first opens and see it in entirely different light.","anticipation",False,"desire to experience same work in altered conditions"),
]},

# ══════════════ FAMILY ══════════════
{"scenario":"family","topic":"estrangement","utterances":[
("Adult_Child","I have not spoken to you in three years and I am only here because the solicitor required it.","anger",False,"forced contact after deliberate estrangement"),
("Parent","I know I have no right to ask anything of you given everything that happened between us.","remorse",False,"parent acknowledges weight of past harm"),
("Adult_Child","The therapist I have worked with for two years told me I did not have to come today at all.","vigilance",False,"self-protective boundary stated clearly"),
("Parent","And yet you came anyway — that matters to me even if it means nothing to you right now.","sadness",False,"reading significance into adult child's presence"),
("Adult_Child","Do not interpret my legal obligation as any form of emotional willingness to be here.","anger",False,"presence clarified as purely legal not personal"),
("Parent","I want you to know that I am sorry for what I failed to protect you from in that house.","remorse",False,"general but genuine apology for parental failure"),
("Adult_Child","Sorry does not restore what those specific years took from me and it never will.","grief",False,"apology understood as insufficient for damage caused"),
("Parent","I know that. I am not asking for forgiveness — only a chance to say it to your face once.","submission",False,"asking nothing, only offering overdue acknowledgement"),
]},
{"scenario":"family","topic":"terminal_illness","utterances":[
("Adult_Child","The oncologist said six to eight months and I have been staring at the wall ever since.","grief",False,"terminal prognosis received for beloved parent"),
("Parent","I need you to hear me — I am at peace with this far more than you are right now.","acceptance",False,"dying parent genuinely at peace with prognosis"),
("Adult_Child","How can you be at peace with six to eight months when I am sitting here losing you.","terror",False,"facing parent's death with overwhelming terror"),
("Parent","Because I have had a full and genuinely good life and you are the best thing in all of it.","joy",False,"profound gratitude for life's fullness"),
("Adult_Child","I am not ready for a world that simply does not have you in it — I never will be.","grief",False,"anticipatory grief over impending irreversible loss"),
("Parent","You are going to be more ready than you think when the time comes — that is the strange truth.","trust",False,"experienced parental wisdom about grief and resilience"),
("Adult_Child","Can we just talk about something completely ordinary today — just be normal for one afternoon.","sadness",False,"craving normalcy before inevitable loss"),
("Parent","Tell me something that made you laugh recently and we will start from there.","serenity",False,"choosing present joy over anticipatory grief"),
]},
{"scenario":"family","topic":"coming_out","utterances":[
("Teen","I am gay and I have known for two years and I cannot keep hiding it from you any longer.","apprehension",False,"coming out after two years of concealment"),
("Mother","You have been carrying this completely alone for two whole years.","sadness",False,"sorrow over child's prolonged solitary burden"),
("Teen","I was scared that you would not be able to look at me the same way after you knew.","terror",False,"fear of parental rejection of fundamental identity"),
("Father","We are going to need a little time to understand what this means fully for our family.","distraction",False,"processing genuinely unexpected information carefully"),
("Teen","I know. I just needed you to actually know because hiding it was making me unwell inside.","acceptance",False,"relief at disclosure despite remaining uncertainty"),
("Mother","We love you. That is the one thing that is not going to need any time at all.","love",False,"unconditional love immediately and clearly affirmed"),
("Teen","I have been so scared of this particular conversation for such a very long time.","fear",False,"years of anticipatory fear finally released"),
("Father","We are glad you trusted us with something this important to who you are.","trust",False,"gratitude for child's confidence in parents"),
]},
{"scenario":"family","topic":"inheritance","utterances":[
("Sibling_1","The will left the entire property to me and nothing to you and I need to understand why.","surprise",False,"unequal inheritance surprising even beneficiary"),
("Sibling_2","I always assumed she held the argument we had five years ago against me permanently.","sadness",False,"estrangement likely caused inheritance exclusion"),
("Sibling_1","I want to split it equally regardless of what the legal document actually says — that is what I want.","trust",False,"fairness chosen clearly over legal entitlement"),
("Sibling_2","You are not legally required to do that and I am absolutely not asking you to do it.","submission",False,"declining to demand what one is not entitled to"),
("Sibling_1","I am not doing this from obligation — I am doing it because we are family and she was wrong.","love",False,"family bond placed decisively above legal document"),
("Sibling_2","She was complicated and she was our mother and I miss her terribly anyway.","grief",False,"grief for flawed but still deeply loved parent"),
("Sibling_1","Same. Let me have my solicitor draw up the paperwork this week and we sort it properly.","trust",False,"committing firmly to equitable action"),
]},
{"scenario":"family","topic":"addiction","utterances":[
("Parent","The police called at two in the morning to say they had found you unresponsive in the street.","terror",False,"child found unconscious due to substance use"),
("Adult_Child","I did not mean for it to reach that point — I need you to know that much at least.","remorse",False,"overdose was not intentional"),
("Parent","It reached exactly as far as it was always going to go without proper intervention.","sadness",False,"trajectory recognised as predictable and preventable"),
("Adult_Child","I am not going to argue with you — you are right about all of it and I am frightened.","fear",False,"own fear acknowledged after near-death experience"),
("Parent","There is a residential programme with an available bed opening on Monday morning.","anticipation",False,"concrete actionable pathway to treatment offered"),
("Adult_Child","I am scared of who I am without the substance because I have not met that person in years.","fear",False,"identity dissolution feared in sobriety"),
("Parent","We will meet that person together. You are not going to figure any of this out alone.","love",False,"unconditional parental love and support offered"),
("Adult_Child","I am deeply sorry for everything I have put you through in all this time.","remorse",False,"genuine guilt over prolonged impact on family"),
]},
{"scenario":"family","topic":"loathing_abuse","utterances":[
("Adult_Child","You treated me with open contempt for twenty years and expected gratitude for the bare minimum.","rage",False,"decades of emotional abuse finally confronted"),
("Parent","I did my best with what I had available and that is what I will always maintain.","distraction",False,"deflecting accountability with hollow statement"),
("Adult_Child","Your best was belittling me in front of my friends every single opportunity you could find.","loathing",False,"specific abuse recalled with intense revulsion"),
("Parent","Children need to be toughened up and I was preparing you for how the real world actually works.","disgust",True,"disgusting rationalisation of systematic abuse"),
("Adult_Child","You have recycled that exact line for thirty years and it still makes me want to leave immediately.","loathing",False,"rationalisation experienced as ongoing profound revulsion"),
("Parent","I see you have been talking to therapists about how terrible I apparently was.","contempt",False,"therapy dismissed with casual contempt"),
("Adult_Child","I am not here expecting an apology because I know I am never going to receive one from you.","acceptance",False,"peace reached without needing parental validation"),
("Adult_Child","I am here to say this once directly to your face so I never have to carry it internally again.","vigilance",False,"confrontation entirely for own closure not theirs"),
]},
{"scenario":"family","topic":"disapproval_career","utterances":[
("Parent","You left a stable medical career to open a bakery and I genuinely need you to help me understand this.","disapproval",False,"parental disapproval of major career change"),
("Adult_Child","The medical career was killing me slowly from the inside and you knew it even if you looked away.","anger",False,"parent's deliberate blindness to suffering called out"),
("Parent","But the financial security and everything we sacrificed to fund that entire education.","sadness",False,"parental sacrifice feeling permanently wasted"),
("Adult_Child","I am grateful for every bit of it and I cannot live that particular life — both things are true.","trust",False,"gratitude and boundary held simultaneously"),
("Parent","I worry that in five years you will look back and feel you threw something irreplaceable away.","apprehension",False,"genuine parental fear for child's long-term future"),
("Adult_Child","And I worry that in five years I will look back and wish I had left a full decade sooner.","optimism",False,"confidence in decision despite ongoing uncertainty"),
("Parent","I do not fully understand it but I suppose I need to trust that you know yourself better than I do.","submission",False,"yielding to adult child's superior self-knowledge"),
("Adult_Child","That is genuinely all I needed to hear from you — thank you for getting there.","love",False,"acceptance felt as profound and generous gift"),
]},
{"scenario":"family","topic":"awe_birth","utterances":[
("New_Parent","I have been present for births professionally before and nothing prepared me for my own child arriving.","awe",False,"first child's birth overwhelming all prior experience"),
("Midwife","First-time parents say exactly that every time — there is simply no preparation for this specific thing.","trust",False,"professional validation of universal parental reaction"),
("New_Parent","She looked directly at me the moment she was placed on my chest and everything shifted.","awe",False,"first eye contact producing immediate overwhelming awe"),
("Partner","I watched him holding her and I had absolutely no words for what I was witnessing.","awe",False,"witnessing partner's first contact producing vicarious awe"),
("New_Parent","I understand everything differently now — every parent I have ever known and every child.","amazement",False,"birth producing fundamental irreversible perspective shift"),
("Midwife","This is the right response — this is exactly what is supposed to happen in this room.","acceptance",False,None),
("Partner","I keep looking at her and thinking we actually made this entirely — completely astonishing.","ecstasy",False,"wonder at having created entirely new human life"),
]},
{"scenario":"family","topic":"sibling_rivalry","utterances":[
("Sibling_A","You always received more attention from both parents growing up and I need to say that finally.","anger",False,"long-held belief about favouritism surfaced"),
("Sibling_B","You actually genuinely believe that — I have the opposite exact experience of our childhood.","amazement",False,"completely opposite perception of same childhood"),
("Sibling_A","Every weekend was about your athletic competitions and every dinner was your achievements.","sadness",False,"feeling consistently overlooked in family narrative"),
("Sibling_B","And every academic prize and university application was entirely about you and your future.","sadness",False,"mirroring exact feeling from opposite perspective"),
("Sibling_A","I had no idea you felt unseen in exactly that way — I genuinely did not know that.","surprise",False,"other's hidden experience genuinely surprising"),
("Sibling_B","We both felt invisible to them in completely different ways at exactly the same time.","grief",False,"parallel loneliness in same family system"),
("Sibling_A","Maybe we should actually talk to them about it together rather than just between ourselves.","anticipation",False,"possibility of family conversation considered"),
("Sibling_B","Or we simply make sure we never repeat that pattern with our own children going forward.","trust",False,"breaking generational pattern as primary goal"),
]},

# ══════════════ ROMANCE ══════════════
{"scenario":"romance","topic":"proposal","utterances":[
("Partner_1","I had an entire speech prepared for this and now you are standing there and every word is gone.","apprehension",False,"prepared speech abandoned in overwhelming moment"),
("Partner_2","You are actually shaking right now — are you genuinely alright.","interest",False,"physical nervousness noticed with concern"),
("Partner_1","I have never been more certain of something and more terrified simultaneously in my entire life.","awe",False,"certainty and vulnerability existing in perfect tension"),
("Partner_2","Just say what is actually in your head — nothing prepared will be better than what is real.","trust",False,"invitation for authentic unscripted expression"),
("Partner_1","I want to build the entire rest of my life alongside you and I am asking if you want the same.","love",False,"direct proposal from authentic unguarded emotion"),
("Partner_2","Yes. Completely and entirely yes. I have been waiting for this specific question for eighteen months.","ecstasy",False,"acceptance with joy and enormous relief combined"),
("Partner_1","Eighteen months and you just let me stand here preparing speeches the whole time.","joy",False,"playful mock complaint from place of happiness"),
("Partner_2","I wanted you to ask in your own time — I was not going to rush what comes next for us.","love",False,"patience as profound expression of love"),
]},
{"scenario":"romance","topic":"awe_connection","utterances":[
("Partner_1","I keep trying to explain what this relationship is like to people and I cannot find adequate words.","awe",False,"depth of connection defying ordinary description"),
("Partner_2","What do you mean when you actually try to describe it to them.","interest",False,None),
("Partner_1","That being with you is like standing somewhere with no visible edge — infinite in every direction.","awe",False,"love described as boundless and without limit"),
("Partner_2","That is simultaneously the most beautiful and slightly unsettling thing anyone has said to me.","amazement",False,"profound compliment received with appropriate wonder"),
("Partner_1","It unsettles me too honestly. I have never felt this completely known by another person.","awe",False,"depth of being known producing genuine wonder"),
("Partner_2","I think what we have is genuinely rare and I try to hold it carefully every single day.","love",False,"conscious daily appreciation of rare connection"),
("Partner_1","I fall a little more in love with you every time you say something like that.","ecstasy",False,"love deepening with each honest exchange"),
]},
{"scenario":"romance","topic":"submission_relationship","utterances":[
("Partner_1","I have accepted that your career opportunity is going to take us to a different country next year.","submission",False,"deferring genuinely to partner's career trajectory"),
("Partner_2","You are not just saying that because you feel you have to say it are you.","interest",False,"checking carefully for genuine acceptance"),
("Partner_1","My honest first instinct was resistance and I worked through it over several weeks.","trust",False,"authentic emotional process shared openly"),
("Partner_2","Tell me what the resistance was actually about because I need to hear the real answer.","interest",False,"requesting genuine feelings not polished response"),
("Partner_1","I was frightened of losing my own sense of place and identity by following someone else's path.","fear",False,"legitimate and real fear about identity loss"),
("Partner_2","That is a completely real fear and I want us to build your anchor in the new place from day one.","love",False,"partner's needs actively centred in the plan"),
("Partner_1","That specific thing is what changed my decision — knowing you had already thought about me.","submission",False,"trust in partner's care enabling genuine acceptance"),
("Partner_2","We are both going to that city together — not you following me but us going.","trust",False,"framing as mutual choice rather than sacrifice"),
]},
{"scenario":"romance","topic":"betrayal","utterances":[
("Partner_1","I read the messages while your phone was on the table and I am not going to pretend I did not.","vigilance",False,"evidence of infidelity discovered and confronted"),
("Partner_2","There is nothing in those messages that is what you are thinking it is right now.","distraction",False,"initial deflection before full admission"),
("Partner_1","I am extremely good at reading and I know precisely what I read on that screen.","anger",False,"dismissal firmly and clearly rejected"),
("Partner_2","I made a catastrophic mistake and I have been trying to find the words to tell you for weeks.","remorse",False,"infidelity finally admitted directly"),
("Partner_1","You let me plan our entire future together while you were hiding this from me — that is the part.","rage",False,"sustained deception during shared planning producing rage"),
("Partner_2","I know. There is genuinely nothing I can say that makes this less than what it is.","sadness",False,"accepting full unmitigated weight of consequences"),
("Partner_1","I need you to go tonight. I simply cannot look at you right now without falling apart.","grief",False,"immediate physical separation needed"),
]},
{"scenario":"romance","topic":"disapproval_family","utterances":[
("Parent","This person is not right for you and I am saying this as someone who loves you without condition.","disapproval",False,"parental disapproval of chosen romantic partner"),
("Adult_Child","This person makes me happy in a way I have genuinely never experienced before in my life.","anger",False,"happiness dismissed by disapproving parent"),
("Parent","Being happy is not the same thing as being with someone stable, kind, or good for you long-term.","disapproval",False,"parental concern about long-term compatibility"),
("Adult_Child","You have met them exactly twice and you have already decided — can you hear how unfair that is.","annoyance",False,"snap judgment on insufficient evidence called out"),
("Parent","My instincts as your parent have been correct about this kind of thing before in your life.","trust",False,"parental authority and track record asserted"),
("Adult_Child","And they have been spectacularly wrong on at least two occasions but let us not open that now.","contempt",True,"past parental errors invoked sarcastically"),
("Parent","I am not asking you to leave them — I am asking you to keep your eyes completely and fully open.","acceptance",False,"conditional acceptance finally offered"),
("Adult_Child","My eyes are completely open. That is precisely how I can see that you are mistaken.","trust",False,"certainty in own judgment asserted firmly"),
]},
{"scenario":"romance","topic":"long_distance","utterances":[
("Partner_1","Long distance is significantly harder than either of us anticipated it would actually be.","grief",False,"distance harder than expected"),
("Partner_2","How are you genuinely coping day to day with it.","interest",False,None),
("Partner_1","I am fine during the week when work keeps me occupied. The weekends are the hardest part.","sadness",False,"weekend emptiness most painful"),
("Partner_2","For me too — Sunday evenings in particular are almost unbearable sometimes.","pensiveness",False,"specific time identified as hardest"),
("Partner_1","Four more months and then this chapter is completely over.","vigilance",False,"counting down to reunion"),
("Partner_2","And then I am there with you permanently — for good and without any more countdowns.","optimism",False,"permanent reunion anticipated with hope"),
("Partner_1","For good. I keep saying that to myself when the evenings get hard.","love",False,"mantra sustaining through difficult periods"),
("Partner_2","We did the hardest part already — this is just the final stretch before everything changes.","trust",False,"encouragement grounded in progress made"),
]},

# ══════════════ SUPPORT ══════════════
{"scenario":"support","topic":"suicidal_ideation","utterances":[
("Friend","You said something last week that I have genuinely not been able to stop thinking about since.","apprehension",False,"concerning statement still resonating"),
("Person","I figured you would ask at some point — I am okay right now in this specific moment.","acceptance",False,"immediate safety confirmed"),
("Friend","What about the moments that are not this particular one right here.","interest",False,"probing about darker periods"),
("Person","There are days when I think very seriously about not being here anymore at all.","grief",False,"suicidal ideation disclosed directly"),
("Friend","Thank you for trusting me with that — it is the most important thing you have ever told me.","love",False,"disclosure received with appropriate gravity"),
("Person","I have never said it to anyone because saying it out loud makes it feel more real.","fear",False,"fear that disclosure validates the thought"),
("Friend","It does not make it more real — it makes it something we can actually address together.","trust",False,"reframing disclosure as protective not confirming"),
("Person","I do not want to die — I just want the pain to stop and I cannot see another way forward.","sadness",False,"suffering driving ideation not actual death wish"),
("Friend","We are going to find another way together — tonight, right now, you and me.","optimism",False,"immediate concrete commitment to joint action"),
]},
{"scenario":"support","topic":"trauma","utterances":[
("Therapist","You mentioned that crowded spaces have become very difficult recently — tell me more about that.","interest",False,"exploring reported specific symptom"),
("Client","My body goes into full alert even in a supermarket — it does not make rational sense to me.","fear",False,"hypervigilance described with confusion"),
("Therapist","It makes complete and perfect sense as a trauma response — your system learned to protect you.","trust",False,"normalising trauma response fully"),
("Client","I feel weak when it happens even though I know intellectually that I should not feel that.","remorse",False,"shame over automatic physiological fear response"),
("Therapist","The shame about the response is often considerably harder to live with than the response itself.","interest",False,"identifying the more painful secondary layer"),
("Client","Yes — the panic lasts five minutes. The shame about the panic lasts all day long.","sadness",False,"shame outlasting the original fear experience"),
("Therapist","We are going to work on both layers in parallel — neither has to precede the other.","optimism",False,"therapeutic approach framed as simultaneous work"),
("Client","I have never told anyone the full version of what happened and I feel lighter already.","acceptance",False,"first full disclosure producing immediate relief"),
]},
{"scenario":"support","topic":"grief_counselling","utterances":[
("Griever","Everyone keeps telling me it gets easier and I am trying very hard to believe them.","apprehension",False,"hope mixed with genuine scepticism"),
("Counsellor","What would easier actually look like to you right now — what would that feel like.","interest",False,"exploring personal definition of recovery"),
("Griever","Being able to think about him without the air completely leaving the room every time.","sadness",False,"grief still physically overwhelming in present"),
("Counsellor","You will reach that. Not by forgetting him but by integrating who he was into who you become.","trust",False,"grief as integration not erasure"),
("Griever","I am terrified of the day it hurts less because that feels like losing him a second time.","terror",False,"fear that healing equals second more permanent loss"),
("Counsellor","That is one of the most profound and insightful things I have heard anyone say about grief.","admiration",False,"deep insight acknowledged and valued"),
("Griever","He would have laughed at me for turning his death into a philosophical problem like this.","joy",False,"personality remembered with genuine warmth"),
("Counsellor","Hold onto that laugh — that is him still present with you in the way that matters.","serenity",False,"presence in living memory validated as real"),
]},
{"scenario":"support","topic":"rage_injustice","utterances":[
("Person","I have been passed over for promotion three cycles while a less experienced colleague was fast-tracked.","rage",False,"systemic racial injustice in career progression"),
("Friend","That is not a coincidence and it is not in your head — I want to say that clearly to you.","trust",False,"validating recognition of real discrimination"),
("Person","My manager told me I am too direct and need to work on how I present myself in meetings.","rage",False,"coded and racially biased feedback received"),
("Friend","Too direct meaning you state facts clearly and expect to be heard as an equal.","contempt",True,"decoding biased feedback with precise sarcasm"),
("Person","I have documented every single instance very carefully for the past eighteen months.","vigilance",False,"systematic documentation of discrimination"),
("Friend","That documentation is your protection and your leverage when this goes further.","trust",False,"advising on legal strategy"),
("Person","The rage is useful but it exhausts me completely — I just want to do my job and be evaluated fairly.","sadness",False,"exhaustion from having to constantly fight bias"),
("Friend","That should be the absolute baseline and the fact that it is not is a systemic failure.","love",False,"solidarity and clear support expressed"),
]},
{"scenario":"support","topic":"loathing_abuse","utterances":[
("Survivor","I stayed for eleven years and I am still asking myself daily what was wrong with me.","remorse",False,"self-blame for staying in abusive relationship"),
("Advocate","Nothing was wrong with you — everything was profoundly wrong with what he did to you.","trust",False,"redirecting blame appropriately and firmly"),
("Survivor","He was completely charming in public and entirely the opposite behind closed doors.","loathing",False,"profound revulsion at abuser's calculated deception"),
("Advocate","That gap between the public and private version is the deliberate weapon — it isolates you completely.","interest",False,"naming the specific mechanism of coercive control"),
("Survivor","I feel a loathing toward him that I did not know I was actually capable of feeling.","loathing",False,"depth of revulsion surprising survivor herself"),
("Advocate","That is healthy and proportionate — your nervous system is recognising the scale of the violation.","trust",False,"validating intensity of emotional response"),
("Survivor","I also feel loathing toward myself and I know rationally that is not fair or right.","loathing",False,"self-directed loathing as consequence of trauma"),
("Advocate","We work specifically on that distinction — his actions versus your response — every session.","optimism",False,"therapeutic separation of responsibility and fault"),
]},
{"scenario":"support","topic":"terror_medical","utterances":[
("Patient","The doctor used the word malignant and everything that came after it was just noise.","terror",False,"cancer diagnosis producing immediate shock"),
("Partner","I was sitting right next to you and I heard it and my whole body went cold instantly.","terror",False,"shared terror in the moment of diagnosis"),
("Patient","I keep replaying that specific moment trying to hear the rest of what he said after it.","distraction",False,"shock preventing information from being retained"),
("Partner","I had my notes app open the entire appointment and wrote down everything he said.","vigilance",False,"practical foresight in preparation for difficult news"),
("Patient","You planned for that possibility before we even walked into that room.","amazement",False,"partner's quiet foresight astonishing in retrospect"),
("Partner","I planned for every version of what that appointment might say because I love you.","love",False,"love expressed through practical invisible preparation"),
("Patient","I am so frightened right now.","terror",False,"raw unguarded fear stated simply"),
("Partner","I know. I am right here with you and I am not going anywhere — not for any of it.","love",False,"unconditional presence pledged without condition"),
]},
{"scenario":"support","topic":"coming_out_support","utterances":[
("Person","I told my parents I am bisexual last night and I cannot quite process how it went.","apprehension",False,"disclosure made and outcome still being processed"),
("Friend","How did it actually go — tell me everything.","interest",False,None),
("Person","Better than I feared and worse than I had hoped — somewhere in the painful middle.","sadness",False,"ambiguous response neither accepting nor rejecting"),
("Friend","The in-between response is its own specific kind of hard to sit with.","acceptance",False,"acknowledging difficulty of non-definitive response"),
("Person","I wanted them to just say they love me no matter what and mean it clearly.","love",False,"longing for unconditional explicit affirmation"),
("Friend","Maybe that comes once they have had time to process what you told them.","optimism",False,"hope that delayed acceptance will arrive"),
("Person","I hope so. I have needed to say this for years and I did it and I feel both lighter and scared.","anticipation",False,"disclosure producing mixed relief and ongoing uncertainty"),
("Friend","You did it. That took real courage and I am enormously proud of you.","admiration",False,"courage in disclosure recognised and celebrated"),
]},

# ══════════════ ACADEMIC ══════════════
{"scenario":"academic","topic":"phd_crisis","utterances":[
("PhD_Student","I am thinking seriously about leaving the doctoral programme this week.","sadness",False,"considering PhD abandonment"),
("Advisor","Tell me more about where this is actually coming from right now.","interest",False,"seeking root cause without judgment"),
("PhD_Student","I have completely lost the thread of why I genuinely care about this research.","grief",False,"motivation for work disappeared entirely"),
("Advisor","That happens to nearly every doctoral student at some point — it does not mean quit.","trust",False,"normalising loss of motivation without minimising"),
("PhD_Student","But what if the motivation simply does not come back no matter what I try.","fear",False,"fear that motivation is permanently gone"),
("Advisor","Take a full week away from the thesis entirely — no reading, no writing, nothing.","acceptance",False,"prescribing strategic distance from work"),
("PhD_Student","That feels deeply counterintuitive but I am willing to trust you on it.","distraction",False,"following advice despite instinctive resistance"),
("Advisor","Sometimes the clearest thinking about work happens at the furthest distance from it.","optimism",False,"reframing absence as productive preparation"),
]},
{"scenario":"academic","topic":"rejection","utterances":[
("Student_A","I was rejected from every single graduate programme I applied to this cycle.","grief",False,"complete rejection from all applications"),
("Student_B","All of them — I am so genuinely sorry.","sadness",False,"empathy for magnitude of disappointment"),
("Student_A","I spent an entire year preparing applications and I believed at least one would say yes.","sadness",False,"year of effort meeting complete rejection"),
("Student_B","That is a profound disappointment and it is completely valid to feel it fully right now.","acceptance",False,"normalising the weight of failure"),
("Student_A","Everything about my future feels genuinely uncertain and foggy right now.","fear",False,"failure producing existential uncertainty"),
("Student_B","One door closed very firmly. That absolutely does not mean there are no other doors.","optimism",False,"reframing without minimising real loss"),
("Student_A","Easy to say when you are not the one staring at every rejection email on your screen.","annoyance",False,"optimism feeling hollow in moment of pain"),
("Student_B","You are right — I am sorry. Take a few days before you think about what comes next.","remorse",False,"acknowledging overreach with appropriate apology"),
]},
{"scenario":"academic","topic":"grade_injustice","utterances":[
("Student","This essay received a failing mark and I want a formal explanation of the specific criteria applied.","anger",False,"grade challenged after apparent injustice"),
("Lecturer","The marking process was conducted blind and returned by an independent external examiner.","trust",False,"independent marking process cited"),
("Student","The feedback says my argument lacks rigour and cites not a single specific example in the text.","contempt",False,"vague feedback insufficient for any learning"),
("Lecturer","That is a fair criticism of the feedback quality and I will raise it directly with the examiner.","acceptance",False,"feedback inadequacy honestly acknowledged"),
("Student","I want to understand precisely which passages triggered that specific judgment.","interest",False,"seeking precise feedback for genuine learning"),
("Lecturer","I will request the fully annotated script so we can review it line by line together.","trust",False,"committing to thorough and transparent review"),
("Student","I am not asking because I want a different grade — I am asking because I need to understand.","vigilance",False,"learning goal clearly distinguished from grade appeal"),
]},
{"scenario":"academic","topic":"boredom_lecture","utterances":[
("Student_1","Three hours of reading identical slides aloud verbatim every single week — genuinely.","boredom",True,"sarcasm about fundamentally dull lecture format"),
("Student_2","The slides are also on the portal so attendance is functionally a performance obligation.","boredom",False,"pointlessness of physical attendance noted"),
("Student_1","I have started bringing knitting because at least my hands are doing something productive.","boredom",False,"coping mechanism for extreme boredom"),
("Student_2","I sit at the back and read the actual textbook during the lecture to compensate.","boredom",False,"self-directed learning replacing passive reception"),
("Student_1","He reads us content we could read ourselves while we pay twenty thousand a year for it.","contempt",False,"fundamental value proposition questioned directly"),
("Student_2","The tutorials are genuinely useful at least — the teaching assistant is exceptionally good.","acceptance",False,"partial positive found within disappointing course"),
("Student_1","She teaches more real content in forty minutes than he covers in three hours.","admiration",False,"striking quality contrast highlighting problem"),
]},
{"scenario":"academic","topic":"first_generation","utterances":[
("Student_1","I am the first person in my entire family to sit in a lecture hall and the imposter feeling is constant.","apprehension",False,"first-gen imposter syndrome pervasive"),
("Student_2","I feel it too and both my parents went to university — imagine how universal this feeling actually is.","amazement",False,"imposter syndrome persisting despite educational lineage"),
("Student_1","When the professor uses references everyone else just nods and I have no idea what they mean.","sadness",False,"cultural capital gap made painful and visible"),
("Student_2","Half of them are nodding to avoid looking lost — do not assume they all actually know.","trust",False,"challenging assumption of others' implicit knowledge"),
("Student_1","I cannot tell my family I find it hard because they sacrificed so much to make this possible.","grief",False,"unable to share struggles due to family expectations"),
("Student_2","That specific loneliness is one of the defining experiences here that never gets spoken about.","sadness",False,"naming the particular isolation of first-gen experience"),
("Student_1","Do you ever wonder whether you genuinely belong in these rooms.","interest",False,"fundamental vulnerability shared aloud"),
("Student_2","Every week. And then something clicks in a seminar and I remember completely why I came.","optimism",False,"moments of genuine validation sustaining presence"),
]},
{"scenario":"academic","topic":"dissertation_crisis","utterances":[
("PhD_Student","My entire methodology assumes stationarity in this dataset and it is demonstrably non-stationary.","terror",False,"fundamental methodological flaw discovered very late"),
("Supervisor","How far through the empirical chapters are you currently.","interest",False,"assessing precise scope of the problem"),
("PhD_Student","Four of five chapters are written entirely around this single assumption.","grief",False,"years of work potentially completely invalidated"),
("Supervisor","This is not fatal — it is serious but it is absolutely not fatal and you need to hear that.","trust",False,"catastrophising corrected without minimising problem"),
("PhD_Student","How is this not fatal when the entire analytical framework rests on this one assumption.","apprehension",False,"unable to see any viable path forward"),
("Supervisor","You reframe non-stationarity as a finding rather than a flaw — it becomes your contribution.","amazement",False,"solution that inverts the problem entirely"),
("PhD_Student","Turn the methodological problem into the novel and unexpected insight.","interest",False,"grasping the approach with growing understanding"),
("Supervisor","Your data is telling you something nobody expected — that is what a doctoral thesis is for.","optimism",False,"unexpected finding reframed as genuine scholarly contribution"),
]},
{"scenario":"academic","topic":"plagiarism_accused","utterances":[
("Professor","The automated similarity report returned a seventy-eight percent match with a published paper.","vigilance",False,"high plagiarism score flagged formally"),
("Student","I cited that paper eleven separate times — the match is because I referenced it very heavily.","trust",False,"similarity explained through extensive citation"),
("Professor","The matched passages appear throughout your arguments, not only in your citation blocks.","apprehension",False,"concern that matching is in reasoning not only references"),
("Student","I built my entire argument in direct response to that paper — that is the thesis structure.","trust",False,"structure explained as sustained dialogue with source"),
("Professor","I need you to walk me through your original contribution section by section with me.","interest",False,"requesting thorough line-by-line review"),
("Student","I can do that right now or in detailed writing — whichever is more useful to you.","vigilance",False,"full cooperation offered without hesitation"),
("Professor","Come to my office tomorrow at ten and bring the complete original draft history.","acceptance",False,None),
("Student","I have version control going back seven months — I will bring the full annotated log.","anticipation",False,"substantial evidence of genuine independent work ready"),
]},

# ══════════════ CONFLICT ══════════════
{"scenario":"conflict","topic":"gaslighting","utterances":[
("Person_A","You told me the meeting was at three and now you are telling me I have the time wrong.","anger",False,"factual dispute about clearly agreed time"),
("Person_B","I said approximately three — there is always an implied margin when I give any time.","distraction",False,"retroactive reframing of clear unambiguous statement"),
("Person_A","You said three o'clock with no qualifier — I wrote it down immediately after you said it.","vigilance",False,"contemporaneous written note cited as evidence"),
("Person_B","You have a consistent habit of hearing what you want and then being absolutely certain of it.","rage",True,"gaslighting through character attack on perception"),
("Person_A","I am not going to argue with my own contemporaneous notes and my own clear memory.","trust",False,"confidence in own perception firmly maintained"),
("Person_B","Perhaps you should seriously consider that your certainty is the pattern here, not my words.","contempt",False,"dismissive attack on reliability of perception"),
("Person_A","I am ending this conversation and returning to it tomorrow with the written exchange included.","vigilance",False,"removing self from manipulative dynamic cleanly"),
]},
{"scenario":"conflict","topic":"loathing_betrayal","utterances":[
("Person_A","You applied for the exact role I told you I was pursuing and hid it from me for two weeks.","rage",False,"betrayal of professional confidence"),
("Person_B","It is a completely competitive process and I had every professional right to apply.","distraction",False,"deflecting with technical correctness"),
("Person_A","I told you in confidence while asking for your specific advice and you used it as intelligence.","loathing",False,"trust deliberately weaponised producing deep revulsion"),
("Person_B","I genuinely believed you would understand that professional ambition is not personal.","distraction",False,"self-interested reframing of deliberate betrayal"),
("Person_A","Using my vulnerability as strategic competitive advantage is about as personal as it gets.","loathing",False,"violation of trust felt as profound and lasting revulsion"),
("Person_B","If you cannot separate professional competition from personal trust we have a separate problem.","contempt",False,"blame shifted entirely to betrayed person"),
("Person_A","We do have a different problem — it is called the permanent end of this relationship.","anger",False,"relationship terminated without qualification"),
]},
{"scenario":"conflict","topic":"rage_discrimination","utterances":[
("Person","I was told I was unsuitable for a customer-facing role specifically because of my accent.","rage",False,"accent discrimination directly experienced in hiring"),
("Manager","That particular language was not used in any feedback that was officially provided.","distraction",False,"denying explicit statement made"),
("Person","The words used were quote cultural fit and communication style in a customer context unquote.","vigilance",False,"coded discriminatory language decoded precisely"),
("Manager","Those are legitimate and uniformly applied business criteria for that specific role.","trust",False,"discriminatory decision defended as neutral policy"),
("Person","Every single person hired has the same regional background as every member of the interview panel.","rage",False,"systematic pattern of discrimination clearly evidenced"),
("Manager","Correlation is not causation and I would strongly caution against that particular framing.","contempt",False,"clear evidence dismissed with technical deflection"),
("Person","I am taking this to an employment tribunal and I have retained every email and every note taken.","aggressiveness",False,"legal action announced with complete evidence ready"),
]},
{"scenario":"conflict","topic":"aggressiveness_negotiation","utterances":[
("Buyer","Your opening price is forty percent above current market rate and I am not here to be insulted.","aggressiveness",False,"hostile rejection of inflated opening offer"),
("Seller","The price reflects proprietary advantages that comparable products simply do not carry.","vigilance",False,"defending premium position with specifics"),
("Buyer","Name every one of them specifically or come down twenty percent immediately — those are the options.","aggressiveness",False,"ultimatum issued with no middle ground"),
("Seller","Proprietary datasets, full regulatory compliance, and six years of client relationship infrastructure.","trust",False,"specific competitive advantages enumerated clearly"),
("Buyer","I have alternatives providing all three at significantly lower cost and I will use them if required.","aggressiveness",False,"leverage made completely explicit"),
("Seller","What number closes this deal today without any further lengthy process.","interest",False,"moving decisively toward resolution"),
("Buyer","Fifteen percent below your current number and I will sign the contract today.","vigilance",False,"final unambiguous position stated"),
("Seller","Ten percent reduction and we include the extended annual support contract at absolutely no cost.","optimism",False,"counter-offer with added value element"),
]},
{"scenario":"conflict","topic":"disapproval_conduct","utterances":[
("Colleague_1","The way you spoke to the junior analyst in that meeting was completely and utterly unacceptable.","disapproval",False,"workplace conduct disapproval voiced directly"),
("Colleague_2","I was under significant pressure and her analysis contained a serious error — I reacted.","distraction",False,"blame shifted to junior employee's mistake"),
("Colleague_1","The analysis had one formatting error and you called her work embarrassing in front of the room.","disapproval",False,"disproportionate and public humiliation called out"),
("Colleague_2","She needs to develop thicker skin if she is going to survive in this particular environment.","contempt",False,"cruel behaviour rationalised as necessary development"),
("Colleague_1","She is twenty-two years old and you are forty-six and you humiliated her publicly.","anger",False,"power imbalance and cruelty named explicitly"),
("Colleague_2","I will speak to her directly and smooth the whole thing over this afternoon.","acceptance",False,"superficial and insufficient resolution offered"),
("Colleague_1","She does not need smoothing over — she needs a genuine apology and it needs to happen today.","disapproval",False,"inadequate response challenged firmly"),
]},
{"scenario":"conflict","topic":"terror_emergency","utterances":[
("Person_A","The smoke alarm has been going for six minutes and I cannot see the stairwell through the smoke.","terror",False,"fire emergency with no visible exit"),
("Person_B","I am on the floor directly below you — the east stairwell is completely clear from here.","trust",False,"safe route communicated precisely"),
("Person_A","I cannot breathe properly and I cannot find the door handle in this smoke.","terror",False,"physical danger escalating rapidly"),
("Person_B","Get as low as possible and move toward the sound of my voice — I am at the stairwell.","vigilance",False,"auditory navigation instructions given"),
("Person_A","I can see your light now — I am moving directly toward it right now.","anticipation",False,"progress toward safety being made"),
("Person_B","You are doing exactly right — I can see you now. Five more meters and you are here.","optimism",False,"encouragement provided during active emergency"),
("Person_A","I am out. I am completely out right now.","ecstasy",False,"safety reached after genuine sustained terror"),
("Person_B","I have got you. Do not try to speak yet — just breathe slowly and steadily.","love",False,"immediate care provided after rescue"),
]},
{"scenario":"conflict","topic":"submission_mediation","utterances":[
("Party_A","I accept the mediator's recommendation even though it is not the outcome I was seeking.","submission",False,"accepting mediated decision against preference"),
("Mediator","Can you say more about what accepting this actually means for you practically going forward.","interest",False,"exploring real practical implications"),
("Party_A","It means changing a process I spent four years building from scratch and that is genuinely hard.","sadness",False,"personal cost of compromise fully acknowledged"),
("Party_B","I want to acknowledge clearly that this resolution required considerably more from your side.","admiration",False,"asymmetry of sacrifice honestly acknowledged"),
("Party_A","It did. But the alternative was a dispute that would have cost both of us far more in the end.","acceptance",False,"pragmatic wisdom in choosing compromise"),
("Mediator","This quality of acceptance is what makes durable resolution genuinely possible.","admiration",False,"facilitator recognising quality of difficult decision"),
("Party_A","I want it recorded that I yielded to the process not because I concede I was wrong.","submission",False,"maintaining position while accepting binding outcome"),
]},
{"scenario":"conflict","topic":"disapproval_policy","utterances":[
("Resident","The proposal to cut the youth centre budget by sixty percent is morally completely indefensible.","disapproval",False,"budget cut strongly and publicly condemned"),
("Councillor","The council faces a thirty million shortfall and all services face proportional reductions.","trust",False,"financial context offered as explanation"),
("Resident","Proportional reductions applied to the least resourced service harm the most vulnerable residents most.","disapproval",False,"proportionality challenged directly on equity grounds"),
("Councillor","Your point regarding distributional impact has been noted formally in the record today.","acceptance",False,"point acknowledged procedurally but insufficiently"),
("Resident","I want considerably more than a notation in a record — I want a genuine justification.","aggressiveness",False,"procedural acknowledgement rejected as insufficient"),
("Councillor","The full impact assessment will be published before the final council vote takes place.","trust",False,"process transparency offered as response"),
("Resident","The vote is in three weeks and the young people most affected have not been consulted once.","disapproval",False,"lack of affected group consultation condemned"),
("Councillor","A community consultation session has been arranged for this coming Tuesday evening.","acceptance",False,"consultation confirmed following pressure"),
]},

# ══════════════ CASUAL ══════════════
{"scenario":"casual","topic":"disgust_food","utterances":[
("Person_A","I just discovered the block of cheese I used for dinner tonight has been in the fridge six months.","disgust",False,"consuming very old cheese belatedly discovered"),
("Person_B","Six months in the fridge and it did not occur to you to check the date before cooking with it.","amazement",False,"disbelief at failure to check expiry"),
("Person_A","It looked completely fine — there was no visible mould which I now understand means nothing.","distraction",False,"false reassurance from superficial appearance"),
("Person_B","How do you actually feel right now physically.","interest",False,None),
("Person_A","Physically fine but morally devastated — that is the only adequate description.","disgust",True,"self-mocking description of cheese aftermath"),
("Person_B","Morally devastated by a cheese incident is genuinely one of the best things you have ever said.","joy",False,"absurdity of situation appreciated"),
("Person_A","I am going to need several days before I can face the dairy section again with any confidence.","disgust",False,"lingering revulsion from incident"),
]},
{"scenario":"casual","topic":"boredom_routine","utterances":[
("Person_A","I have eaten the exact same lunch every single day for fourteen months and noticed today for the first time.","boredom",False,"profound unconscious monotony suddenly perceived"),
("Person_B","What precisely broke the trance on day four hundred and twenty something.","interest",True,"mock precision of boredom duration"),
("Person_A","Someone sat across from me eating something colourful and eating it with genuine visible enthusiasm.","distraction",False,"contrast with engaged person breaking spell"),
("Person_B","You are describing a person eating a sandwich as if it were an epiphany of some significance.","joy",False,"amusing overstatement of mundane contrast"),
("Person_A","It was an epiphany — I am in a completely beige lunch loop and I had not seen it until today.","boredom",False,"mundane repetition suddenly and fully visible"),
("Person_B","What would you actually eat if lunch had no constraints or habits governing it at all.","interest",False,"prompting imagination beyond current limits"),
("Person_A","I genuinely cannot picture it — the habit has completely replaced the underlying preference.","boredom",False,"personal preference entirely eroded by routine"),
("Person_B","That is simultaneously the saddest and funniest thing you have said all year.","joy",False,"appropriate warm response to absurdity"),
]},
{"scenario":"casual","topic":"terror_accident","utterances":[
("Person_A","The car in front just stopped on the motorway with zero warning and I missed it by half a second.","terror",False,"near-fatal motorway accident"),
("Person_B","Are you okay — are you actually pulled over somewhere safe right now.","apprehension",False,"immediate safety concern"),
("Person_A","I am parked on the hard shoulder shaking and I cannot make my hands stop trembling.","terror",False,"physiological terror response continuing"),
("Person_B","That is your body processing adrenaline — it is doing exactly what it is designed to do.","trust",False,"normalising physical trauma response"),
("Person_A","I keep seeing the gap closing in my head and I cannot make it stop replaying on loop.","terror",False,"intrusive flashback of near-miss continuing"),
("Person_B","Do not attempt to drive again until the shaking stops completely — at least thirty minutes.","vigilance",False,"firm safety instruction given"),
("Person_A","I have never felt that close to something ending — it is a very specific kind of frightening.","awe",False,"proximity to death producing terror-awe blend"),
("Person_B","I am so glad you have thirty minutes sitting safely rather than the alternative.","love",False,"profound gratitude for friend's survival"),
]},
{"scenario":"casual","topic":"disgust_behaviour","utterances":[
("Person_A","He explained her own published research back to her for twenty uninterrupted minutes at the event.","disgust",False,"witnessing condescending gendered behaviour"),
("Person_B","With her standing directly in front of him holding her own published paper presumably.","contempt",True,"sarcasm amplifying absurdity of situation"),
("Person_A","She has a PhD in the precise subject he was explaining to her and getting wrong.","rage",False,"incompetent condescension directed at acknowledged expert"),
("Person_B","Please tell me she corrected him at some point during those twenty minutes.","interest",False,"hoping for satisfying ending to story"),
("Person_A","She waited until he completely finished speaking and then cited her own paper title and year.","ecstasy",False,"expert reclaiming expertise with elegant precision"),
("Person_B","That is the correct response and I am enormously glad she delivered it.","admiration",False,"expert's response fully appreciated"),
("Person_A","The exact moment he realised what had happened — genuinely worth the preceding twenty minutes.","joy",False,"humiliating realisation observed with satisfaction"),
]},
{"scenario":"casual","topic":"submission_game","utterances":[
("Player_1","I completely accept defeat — your strategy in the final round was objectively better than mine.","submission",False,"graceful and clean acceptance of loss"),
("Player_2","You pushed me to my absolute limit twice — I genuinely thought you had me in the middle sequence.","admiration",False,"acknowledging opponent's real strength during match"),
("Player_1","I made one critical error on move thirty-eight and never fully recovered the position after it.","remorse",False,"specific mistake identified and owned"),
("Player_2","That error was subtle — most players would not have seen the downstream consequence of it.","trust",False,"quality of the mistake itself acknowledged"),
("Player_1","I yielded to a genuinely better approach rather than a lucky one and I can learn from that.","submission",False,"defeat reframed as valuable educational experience"),
("Player_2","That precise attitude is why you will be considerably harder to beat next time we play.","admiration",False,"graciousness in defeat recognised as competitive strength"),
("Player_1","Same time next week and I will have specifically worked on that particular sequence.","anticipation",False,"immediate commitment to return, improve and compete"),
]},
{"scenario":"casual","topic":"awe_universe","utterances":[
("Person_1","I spent three hours on the roof last night with a telescope and I have not fully recovered.","awe",False,"extended stargazing producing profound lasting effect"),
("Person_2","What specifically did you see that did this to you.","interest",False,None),
("Person_1","Saturn's rings with my own eyes through glass I own — not a photograph but actual photons.","awe",False,"direct unmediated contact with cosmic scale"),
("Person_2","The actual photons that left that planet — that is what you are saying to me.","amazement",False,"physical reality of light travel fully grasped"),
("Person_1","Eighty minutes of light travel arriving directly into my eye at eleven on a Tuesday evening.","awe",False,"cosmic scale rendered in deliberately mundane terms"),
("Person_2","I need you to stop because I am experiencing vertigo about the concept of a Tuesday now.","awe",False,"borrowed awe from description producing real response"),
("Person_1","Welcome to exactly where I have been since midnight — enjoy the vertigo.","serenity",False,"sharing lingering cosmic perspective with warmth"),
]},
{"scenario":"casual","topic":"boredom_meeting","utterances":[
("Colleague_1","This meeting could absolutely have been an email and has been demonstrating that for forty minutes.","boredom",True,"classic meeting inefficiency complaint with sarcasm"),
("Colleague_2","The agenda has three clear items and we are still on the preamble to the very first one.","boredom",False,"meeting inefficiency observed with precision"),
("Colleague_1","I have written and deleted the same email draft twice while nodding in apparent agreement.","boredom",True,"productive multitasking to survive boredom"),
("Colleague_2","Has he asked anyone in this room a single direct question yet in forty minutes.","distraction",False,"noting complete absence of participant engagement"),
("Colleague_1","Once — he asked if anyone had initial thoughts and then immediately answered his own question.","contempt",True,"rhetorical question identified and called out"),
("Colleague_2","I brought a full packet of biscuits and I am rationing them carefully to last until freedom.","boredom",False,"survival strategy for extended pointless meeting"),
("Colleague_1","That is advanced and experienced meeting preparation and I respect it completely.","admiration",False,"practical foresight warmly appreciated"),
]},
{"scenario":"casual","topic":"disapproval_social","utterances":[
("Person_A","I had to say something when he made those comments at the dinner table last night.","disapproval",False,"challenging behaviour at social gathering"),
("Person_B","What exactly did he say — give me the precise version.","interest",False,None),
("Person_A","He made three separate jokes using disability as the punchline and laughed hardest each time.","disgust",False,"ableist humour observed with revulsion"),
("Person_B","In front of a person with a disability who was actually sitting at the same table.","amazement",False,"complete obliviousness to impact noted with disbelief"),
("Person_A","She went completely quiet and he did not register it once — entirely unaware.","sadness",False,"victim's visible pain invisible to perpetrator"),
("Person_B","What did you actually say when you stepped in to address it.","interest",False,None),
("Person_A","I said those jokes land very differently than he thinks and asked him to please move on.","disapproval",False,"challenge issued calmly but clearly"),
("Person_B","That was exactly the right call and it required real courage to be the one to say it.","admiration",False,"intervention recognised as genuinely courageous"),
]},
{"scenario":"casual","topic":"awe_art","utterances":[
("Visitor_1","I have walked past reproductions of this painting my entire life and the original is nothing like them.","awe",False,"first encounter with original versus reproduction"),
("Visitor_2","The physical scale is the first thing — no reproduction communicates how large it actually is.","amazement",False,"physical scale producing immediate wonder"),
("Visitor_1","There is something breathing in the actual brushwork that I simply cannot explain rationally.","awe",False,"physical proximity revealing intangible aliveness"),
("Visitor_1","I think this is what they mean when people say great art is alive — it is actively doing something.","awe",False,"understanding art's aliveness through direct experience"),
("Visitor_2","I want to come back tomorrow morning when the galleries first open and experience it in different light.","anticipation",False,"desire to encounter same work under different conditions"),
]},
{"scenario":"casual","topic":"rage_landlord","utterances":[
("Person_A","The landlord entered my flat without twenty-four hours notice for the fourth time this year.","rage",False,"tenant rights repeatedly and knowingly violated"),
("Person_B","That is illegal under the tenancy agreement and specifically under the Housing Act.","vigilance",False,"legal violation clearly confirmed"),
("Person_A","I know it is illegal and he knows it and he does it anyway because he expects no consequences.","rage",False,"deliberate violation with expectation of impunity"),
("Person_B","The only thing that changes this pattern is making it genuinely costly for him to continue.","interest",False,"practical advice on effective enforcement"),
("Person_A","He is counting entirely on me not wanting the administrative hassle of a formal complaint.","contempt",False,"power dynamic and his strategy clearly identified"),
("Person_B","File the formal complaint immediately — environmental health and the deposit protection scheme.","vigilance",False,"specific actionable steps provided"),
("Person_A","I am so tired of having to fight aggressively for rights I am legally entitled to already.","sadness",False,"exhaustion from constant necessary advocacy"),
("Person_B","File it anyway — one formal response typically ends this pattern permanently.","optimism",False,"encouragement grounded in realistic expectation"),
]},

# ══════════════ SOCIAL ══════════════
{"scenario":"social","topic":"awe_performance","utterances":[
("Audience_1","I have attended dozens of concerts in my life and nothing in my memory compares to tonight.","awe",False,"extraordinary live performance exceeding all prior experience"),
("Audience_2","She held that final note and the entire venue fell completely silent for three full seconds.","amazement",False,"post-performance silence from collective awe"),
("Audience_1","Tears were running down my face before I even realised I was crying at all.","awe",False,"involuntary emotional response to transcendent performance"),
("Audience_2","The woman sitting next to me grabbed my arm at the end of the second movement.","amazement",False,"collective physical response to overwhelming music"),
("Audience_1","I want to describe what made it this extraordinary but language keeps completely failing me.","distraction",False,"art exceeding ordinary descriptive capacity"),
("Audience_2","This is the kind of evening you remember clearly at the very end of your life.","awe",False,"recognising once-in-a-lifetime quality of experience"),
("Audience_1","I do not want to analyse it too much in case the analysis ruins what I am still feeling.","serenity",False,"protecting emotional experience from premature intellectualisation"),
]},
{"scenario":"social","topic":"disapproval_event","utterances":[
("Guest_1","The groom's speech contained three separate jokes about his wife not being allowed to have opinions.","disapproval",False,"sexist wedding speech content observed"),
("Guest_2","In front of two hundred guests and he thought it was charming and original.","disgust",False,"misogyny deployed in very public setting"),
("Guest_1","Her face during the third joke was the most controlled and painful thing I have witnessed.","sadness",False,"bride's managed pain visible only to careful observers"),
("Guest_2","I looked toward her mother's table and there was a very pointed and deliberate silence.","interest",False,"family's response registered"),
("Guest_1","Someone should have said something clearly before it reached the third joke.","remorse",False,"regret over collective group silence"),
("Guest_2","I was the fourth person I saw notice and none of us moved and that makes me part of it.","remorse",False,"accountability for passive witnessing accepted"),
("Guest_1","Collective silence sends its own message to the person at the centre of the room.","disapproval",False,"impact of group inaction recognised and named"),
]},
{"scenario":"social","topic":"submission_authority","utterances":[
("Junior_Officer","I disagree with this specific order but I understand I am required to follow it professionally.","submission",False,"professional deference despite clear personal disagreement"),
("Senior_Officer","Note your formal objection on the record and then carry out the instruction as given.","trust",False,"disagreement channelled through correct official process"),
("Junior_Officer","I want my objection clearly on the official record before I proceed with this.","vigilance",False,"accountability trail deliberately established"),
("Senior_Officer","It is noted and formally documented — proceed when you are ready.","acceptance",False,None),
("Junior_Officer","This situation represents precisely why I believe this protocol needs formal review.","aggressiveness",False,"using outcome to push for necessary policy change"),
("Senior_Officer","Submit that formal recommendation in writing once the task is complete and I will support it.","trust",False,"supporting proper channel challenge"),
("Junior_Officer","Understood. Proceeding now under my formally stated objection.","submission",False,"complying while preserving and maintaining dissent"),
]},
{"scenario":"social","topic":"awe_universe_social","utterances":[
("Person_1","The James Webb images released today show galaxies forming just four hundred million years after the Big Bang.","awe",False,"cosmic scale imagery producing overwhelming wonder"),
("Person_2","Four hundred million years after the beginning of all of existence and we are looking at it now.","amazement",False,"temporal scale fully grasped producing amazement"),
("Person_1","Light that left those galaxies over thirteen billion years ago is arriving right now at a telescope.","awe",False,"physical reality of cosmic time producing awe"),
("Person_2","And being transformed into an image that I am looking at on my phone while eating lunch.","amazement",False,"absurd juxtaposition of cosmic and mundane"),
("Person_1","The contrast between the scale of what you are looking at and the screen you are looking at it on.","awe",False,"medium-message contrast producing wonder"),
("Person_2","I feel simultaneously very small and extraordinarily privileged to be alive in this particular moment.","awe",False,"temporal privilege of witnessing recognised"),
("Person_1","This is one of those images that changes how you understand where you actually are in the universe.","awe",False,"perspective-shifting encounter with cosmic scale"),
]},

# ══════════════ WELLBEING ══════════════
{"scenario":"wellbeing","topic":"disgust_unhealthy","utterances":[
("Person_A","I ate fast food four consecutive days this week and I feel physically disgusted with myself.","disgust",False,"self-directed disgust from sustained poor diet"),
("Person_B","Disgust with yourself is genuinely not a good starting point for building a healthier habit.","trust",False,"challenging shame-based motivation directly"),
("Person_A","Then what is a better starting point because shame is currently the only engine I have.","interest",False,"genuine question about alternative motivation"),
("Person_B","Curiosity about how you actually feel when you eat differently rather than guilt when you do not.","interest",False,"curiosity-based approach proposed as replacement"),
("Person_A","I have honestly forgotten what I feel like when I eat well — it has genuinely been that long.","sadness",False,"healthy baseline completely forgotten"),
("Person_B","One week of just noticing how food makes you feel rather than judging the choice itself.","optimism",False,"low-stakes experiment proposed with no pressure"),
("Person_A","Noticing instead of judging sounds like something an extremely zen person came up with.","joy",True,"gentle affectionate mockery of therapeutic framing"),
("Person_B","Correct — and it works anyway regardless of the source.","acceptance",False,None),
]},
{"scenario":"wellbeing","topic":"terror_health","utterances":[
("Patient","The heart palpitations have been happening every single day for two weeks before I told anyone.","apprehension",False,"delaying disclosure of concerning physical symptom"),
("Doctor","What made you wait a full two weeks before coming in to see someone about it.","interest",False,"exploring delay in seeking help"),
("Patient","I was terrified of what it might actually mean and I thought ignoring it might make it go away.","terror",False,"avoidance driven entirely by fear of diagnosis"),
("Doctor","That is one of the most common reasons people delay seeking help — fear of confirmation.","acceptance",False,"normalising avoidance behaviour without judgment"),
("Patient","If there is something seriously wrong I have a six-year-old at home — that is what I kept thinking.","grief",False,"parental fear driving health anxiety specifically"),
("Doctor","Let us find out what is actually happening before we start imagining scenarios.","trust",False,"redirecting from catastrophising to present reality"),
("Patient","I know I should have come sooner. I am just scared to know and more scared not to.","terror",False,"dilemma between knowledge and ignorance named"),
("Doctor","That is the bravest possible reason to be sitting in this specific chair today.","admiration",False,"courage in seeking help recognised and affirmed"),
]},
{"scenario":"wellbeing","topic":"rage_system","utterances":[
("Person","I have been waiting eleven months for a mental health referral and was just told another eight months.","rage",False,"systemic mental healthcare failure and injustice"),
("Advisor","That wait time is completely unacceptable and I want to help you explore all available options.","trust",False,"immediate engagement with individual's real problem"),
("Person","I have deteriorated significantly in eleven months while waiting for help with the deterioration itself.","rage",False,"Kafkaesque system failure named precisely"),
("Advisor","Have you accessed any crisis support services at all during this waiting period.","interest",False,"checking emergency provision utilisation"),
("Person","Three times — including once when I was seriously considering not being here anymore.","grief",False,"crisis reached during systemic failure"),
("Advisor","I am escalating this case to the urgent pathway today — eleven months qualifies you.","vigilance",False,"advocate taking immediate concrete action"),
("Person","The rage has become its own separate exhaustion on top of everything else.","sadness",False,"rage consuming limited emotional resources"),
("Advisor","That rage is completely rational and it deserves to have a useful purpose now.","admiration",False,"channelling justified anger productively"),
]},
{"scenario":"wellbeing","topic":"boredom_recovery","utterances":[
("Person","I am six weeks post-surgery and the boredom has itself become a serious symptom.","boredom",False,"recovery boredom worse than expected physical pain"),
("Friend","More profoundly bored than week two when you genuinely could not move at all.","interest",False,"seeking timeline context"),
("Person","Week two had its own particular horror — week six has physical recovery but zero stimulation.","boredom",False,"different recovery phase presenting different problem"),
("Friend","What can you actually do now that you physically could not do in week two.","interest",False,"assessing current functional capabilities"),
("Person","Sit upright for extended periods. Type messages. Stare at things with marginally better posture.","boredom",True,"sardonic description of severely limited capability"),
("Friend","I am coming over Saturday with a two-thousand-piece puzzle and absolutely no mercy for weakness.","anticipation",False,"active concrete plan to fight debilitating boredom"),
("Person","A two-thousand-piece puzzle sounds like a genuine lifeline and I am not being dramatic about that.","joy",False,"relief at planned distraction enormous and real"),
]},
{"scenario":"wellbeing","topic":"loathing_self","utterances":[
("Person","I look in the mirror and feel a revulsion I cannot explain to anyone who has not felt it themselves.","loathing",False,"profound body-directed self-loathing"),
("Therapist","That feeling has a name and it is not unique to you even though it feels completely isolating.","trust",False,"normalising extreme self-directed feeling"),
("Person","I know intellectually that what I see is distorted but the revulsion itself is not intellectual.","distraction",False,"gap between cognitive knowledge and emotional experience"),
("Therapist","That precise gap between what you know and what you feel is exactly where our work happens.","interest",False,"naming the specific therapeutic territory"),
("Person","I have felt this since I was twelve years old and I am thirty-one now — nineteen years of this.","grief",False,"duration of suffering quantified with precision"),
("Therapist","Nineteen years is a very long time to carry something that was never actually yours to carry.","sadness",False,"duration put in perspective of origin"),
("Person","Whose was it then because I am the one who has to live inside this body every single day.","anger",False,"frustrated question about origin and responsibility"),
("Therapist","That is exactly the right question and answering it together is the work we are here for.","optimism",False,"framing therapy as collaborative process of discovery"),
]},
{"scenario":"wellbeing","topic":"sober_anniversary","utterances":[
("Person","I have been completely sober for one full year as of this morning.","ecstasy",False,"one year sobriety milestone reached"),
("Friend","One complete year — that is genuinely extraordinary and you should know that.","admiration",False,"milestone recognised as significant achievement"),
("Person","There were days in the first month when I did not believe I would make it to week four.","grief",False,"early difficulty remembered with honesty"),
("Friend","And here you are twelve months later standing in front of me.","amazement",False,"progress made visible and real"),
("Person","It was not linear — there were some hard weeks in month seven.","remorse",False,"honest acknowledgement of difficult periods"),
("Friend","And you got back up every single time those weeks happened.","admiration",False,"resilience across full year recognised"),
("Person","I am actually proud of myself and I have not felt that in a very long time.","joy",False,"self-pride returned after years of absence"),
("Friend","You absolutely should be — genuinely and without any qualification.","love",False,"unconditional affirmation of earned pride"),
]},

# ══════════════ COMMUNITY ══════════════
{"scenario":"community","topic":"aggressiveness_rights","utterances":[
("Resident_1","They approved the development that will eliminate the last green space in this entire district.","rage",False,"planning decision destroying irreplaceable community resource"),
("Resident_2","We submitted over four hundred formal objections and not one was addressed in the council meeting.","rage",False,"community voice systematically and completely ignored"),
("Resident_1","I am going to the council offices tomorrow and I am not leaving without a formal written response.","aggressiveness",False,"confrontational approach planned in response"),
("Resident_2","I will be there with you. We bring the full petition and we bring cameras for documentation.","aggressiveness",False,"coordinated confrontational action planned together"),
("Resident_1","They are counting on us writing polite letters and going home — we stop doing what they expect.","vigilance",False,"identifying and actively disrupting compliance pattern"),
("Resident_2","The local newspaper has already asked for a statement — this needs to go fully public.","anticipation",False,"media strategy as escalation tool"),
("Resident_1","I am not backing down on this one — not this time and not on this specific space.","aggressiveness",False,"firm commitment to sustained aggressive advocacy"),
]},
{"scenario":"community","topic":"elder_care","utterances":[
("Resident_A","The elderly man at the end of the road has no family nearby and I have been worried.","sadness",False,"concern for isolated elderly neighbour"),
("Resident_B","I have noticed his curtains stay closed much later than they used to on some days.","interest",False,"subtle sign of possible deterioration noticed"),
("Resident_A","I started leaving his shopping on his doorstep when I do my own every Thursday.","love",False,"quiet practical care extended without being asked"),
("Resident_B","He must appreciate that gesture enormously even if he cannot easily express it.","admiration",False,"quiet care recognised as meaningful"),
("Resident_A","He left a handwritten note yesterday saying I remind him of his daughter who lives abroad.","love",False,"profound emotional impact of simple gesture"),
("Resident_B","That is one of the genuinely nicest things I have heard anyone say in a very long time.","joy",False,"warmth at hearing story of connection"),
("Resident_A","I should have done it years ago — he has been alone on that street for a long time.","remorse",False,"regret over delayed action"),
("Resident_B","You are doing it now and that is what matters to him on each Thursday.","acceptance",False,"present action more important than past delay"),
]},
{"scenario":"community","topic":"awe_community","utterances":[
("Volunteer_1","Seven hundred people showed up to rebuild the community hall in a single weekend.","amazement",False,"extraordinary scale of spontaneous collective action"),
("Volunteer_2","People were arriving before six in the morning with tools and food and no prior coordination.","awe",False,"self-organising community action producing wonder"),
("Volunteer_1","I have lived here twenty years and I did not know this capacity existed in this neighbourhood.","awe",False,"latent community strength suddenly revealed"),
("Volunteer_2","By Sunday afternoon the roof was completely done and the kitchen was already functioning.","amazement",False,"speed and effectiveness of collective effort"),
("Volunteer_1","Someone brought a generator and someone else brought a sound system and it just happened.","awe",False,"collective self-organisation producing awe"),
("Volunteer_2","I felt genuinely proud to live here in a way I have not felt for quite a long time.","joy",False,"belonging and civic pride renewed by event"),
("Volunteer_1","That weekend changed something about how I understand what this place actually is.","awe",False,"community's character fundamentally and positively reappraised"),
]},

# ══════════════ TRAVEL ══════════════
{"scenario":"travel","topic":"awe_landscape","utterances":[
("Traveller_1","We turned a corner in Iceland and the Northern Lights were simply there with no warning at all.","awe",False,"unexpected encounter with Northern Lights"),
("Traveller_2","What does it actually look like standing underneath them — photographs make it seem unreal.","interest",False,"curiosity about direct unmediated experience"),
("Traveller_1","It looks like someone drew directly onto the sky in moving light — photographs are simply not wrong.","awe",False,"phenomenon described as beautiful impossibility"),
("Traveller_2","Did it make any sound or was it only light with no accompanying sound at all.","interest",False,"seeking specific sensory detail"),
("Traveller_1","Complete and total silence underneath them — which somehow made it more overwhelming not less.","awe",False,"silence amplifying rather than diminishing visual wonder"),
("Traveller_2","I sat down directly on the snow because my legs stopped working reliably.","amazement",False,"physical response to overwhelming stimulus"),
("Traveller_1","I cried and could not tell you exactly why — it was not sad in any conventional sense.","awe",False,"emotion too large for ordinary categorical labels"),
("Traveller_2","Maybe that is precisely what awe actually is — emotion that exceeds the usual labels entirely.","awe",False,"attempting to define the experience through experience"),
]},
{"scenario":"travel","topic":"terror_travel","utterances":[
("Traveller_1","The boat engine cut out completely twenty miles offshore with absolutely no phone signal.","terror",False,"ocean breakdown with complete communication failure"),
("Traveller_2","How many people were on the boat and what were the sea conditions when it happened.","interest",False,"assessing severity of situation"),
("Traveller_1","Four of us and the sea was becoming noticeably rougher approximately every twenty minutes.","terror",False,"conditions deteriorating while stranded"),
("Traveller_2","Was there anyone in the group who managed to maintain composure through it.","interest",False,"asking about group dynamics under pressure"),
("Traveller_1","The guide was completely extraordinary — calm and systematic while the rest of us were not.","admiration",False,"professional calm admired under genuine danger"),
("Traveller_2","What eventually got you all back to shore safely.","interest",False,None),
("Traveller_1","A passing fishing vessel spotted our signal mirror at dusk — one hour before complete darkness.","amazement",False,"rescue by fortunate chance at critical last moment"),
("Traveller_2","I feel a visceral response just listening to this and I was not even present for it.","awe",False,"empathetic awe at survival account"),
]},
{"scenario":"travel","topic":"submission_culture","utterances":[
("Traveller","I did not understand the specific custom but I followed my host's guidance without any question.","submission",False,"cultural deference in genuinely unfamiliar context"),
("Friend","What was the custom that you were following so carefully.","interest",False,None),
("Traveller","You remove your shoes and accept tea before any conversation of real substance begins.","acceptance",False,"cultural norm described with respect"),
("Friend","And did that feel natural to you or did you have to consciously work to adapt to it.","interest",False,"exploring quality of cultural adaptation"),
("Traveller","Entirely conscious — every instinct said move to the point but I submitted to their pace.","submission",False,"deliberate suppression of own cultural defaults"),
("Friend","What happened when you genuinely let go of your urgency and followed their rhythm.","interest",False,None),
("Traveller","The conversation we eventually had was the most substantive I have ever had with a stranger.","amazement",False,"yielding producing significantly better outcome than rushing"),
("Friend","Your submission to their rhythm produced something your rhythm never would have generated.","trust",False,"insight into concrete value of cultural deference"),
]},
{"scenario":"travel","topic":"disapproval_tourism","utterances":[
("Traveller_1","I watched forty tourists photograph a sacred site while someone was actively praying there.","disapproval",False,"tourists disrespecting active place of worship"),
("Traveller_2","Did anyone from the group say anything at all to them about it.","interest",False,None),
("Traveller_1","A local guide asked them quietly to stop and three of them turned their cameras on him.","loathing",False,"redirected violation producing worse secondary violation"),
("Traveller_2","Using the camera to document being challenged rather than to comply — that is a specific awful.","disgust",False,"weaponising documentation against protector"),
("Traveller_1","I have been to many countries and that was the most openly disrespectful thing I have witnessed.","disapproval",False,"behaviour judged against broad comparative standard"),
("Traveller_2","Tourism without genuine consent to the culture it visits is simply extraction with a camera.","contempt",False,"structural critique of exploitative tourism"),
("Traveller_1","I wrote a formal complaint to the tour operator but my expectations for response are very low.","sadness",False,"formal action taken with realistic low expectations"),
]},

# ══════════════ TECHNOLOGY ══════════════
{"scenario":"technology","topic":"aggressiveness_tech","utterances":[
("Developer_2","That is the only metric they actually measure — you are making precisely the right decision.","trust",False,"validation of aggressive market response"),
]},
# ══════════════ ADDITIONAL DATA FOR CONSTRAINT SATISFACTION ══════════════
{"scenario":"workplace","topic":"sarcasm_efficiency","utterances":[
("Manager","I have scheduled a pre-meeting to discuss the agenda for our actual strategy meeting tomorrow.","boredom",False,"excessive meetings scheduled"),
("Employee","Oh, fantastic — because what my calendar really lacked was a nested hierarchy of administrative overhead.","contempt",True,"sarcastic response to meeting about a meeting"),
("Manager","We need to ensure every stakeholder is perfectly aligned before we open the main discussion.","trust",False,"manager justifying the extra meeting"),
("Employee","I am sure the three hours of collective productivity we are losing will be well worth the alignment.","disapproval",True,"sarcastic disapproval of productivity loss"),
("Manager","I appreciate your commitment to the process — let us make it a productive session.","acceptance",False,None),
("Employee","I will bring my enthusiasm and a very large coffee to sustain the illusion of engagement.","contempt",True,"sarcastic contempt for the process"),
]},
{"scenario":"conflict","topic":"sarcasm_accountability","utterances":[
("Person_A","I forgot to mention that I used your car and accidentally scraped the entire passenger side door.","remorse",False,"accidental damage to property admitted"),
("Person_B","Oh, do not worry about it — I was actually thinking the paint looked far too consistent anyway.","rage",True,"sarcastic rage over car damage"),
("Person_A","I am genuinely sorry and I am going to pay for the full repair as soon as possible.","remorse",False,"sincere apology and offer to pay"),
("Person_B","A repair? Why stop there? Maybe we should just scrape the other side so it matches.","anger",True,"sarcastic anger suggesting more damage"),
("Person_A","I understand you are angry and I am not trying to minimize what happened here at all.","submission",False,"yielding to justified anger"),
("Person_B","Your understanding is a true comfort while I stare at fifteen hundred dollars of body work.","contempt",True,"sarcastic contempt for the apology"),
]},
{"scenario":"social","topic":"sarcasm_pretension","utterances":[
("Guest_1","He spent the entire dinner party explaining the subtle notes of a wine he bought for six dollars.","disgust",False,"witnessing pretentious behaviour"),
("Guest_2","A truly vintage performance — I particularly enjoyed the part where he described the 'hint of regret'.","contempt",True,"sarcastic mockery of pretension"),
("Guest_1","He actually corrected the host on the proper temperature for serving a basic table red.","disapproval",False,"disapproval of guest's rudeness"),
("Guest_2","His social grace is as refined as his palate — it is genuinely a privilege to be in his orbit.","contempt",True,"sarcastic contempt for the guest's social skills"),
("Guest_1","I think I am going to 'accidentally' spill my next drink if he starts talking about terroir again.","aggressiveness",False,"planned aggressive intervention"),
("Guest_2","Please do — I will document the tragic loss of such a 'complex' vintage for posterity.","joy",True,"sarcastic joy at the prospect of the spill"),
]},
{"scenario":"romance","topic":"sarcasm_chores","utterances":[
("Partner_1","I noticed you left your wet towels on the bed for the third time this week after promising not to.","annoyance",False,"repeated neglect of shared chores"),
("Partner_2","I was performing a scientific experiment to see how much mildew the mattress could actually absorb.","contempt",True,"sarcastic deflection of chore responsibility"),
("Partner_1","I am so glad our home has become a laboratory for your fascinating research into household decay.","disapproval",True,"sarcastic disapproval of partner's behaviour"),
("Partner_2","I do it all for you, darling — I know how much you value a stimulating environment.","love",True,"sarcastic love used to deflect conflict"),
("Partner_1","I would value a dry bed considerably more than I value your contribution to science today.","sadness",False,"expression of genuine dissatisfaction"),
("Partner_2","Fine. I will end the experiment and move the towels — your lack of vision is truly disappointing.","submission",True,"sarcastic submission while complying"),
]},
{"scenario":"workplace","topic":"sarcasm_promotion","utterances":[
("Employee","I heard that Dave got the promotion because he plays golf with the regional director every Sunday.","rage",False,"unfair promotion based on networking"),
("Colleague","Well, obviously — we all know that a strong swing is the most critical metric for software architecture.","contempt",True,"sarcastic mockery of unfair promotion criteria"),
("Employee","I have been leading the core migration for six months and Dave cannot even explain what a container is.","anger",False,"frustration over technical incompetence of promotee"),
("Colleague","Dave provides the 'vision' and the 'cultural glue' — and apparently the 'perfect putt'.","disapproval",True,"sarcastic disapproval of non-meritocratic promotion"),
("Employee","I am going to start carrying my laptop in a golf bag and see if my career trajectory improves.","aggressiveness",True,"sarcastic aggressive plan for career growth"),
("Colleague","I will be your caddy — we can discuss database sharding between the eighth and ninth holes.","joy",True,"sarcastic participation in the mockery"),
]},
{"scenario":"conflict","topic":"sarcasm_gaslighting","utterances":[
("Person_A","You said you would be here at six and it is now nearly eight and you haven't even apologized.","rage",False,"significant lateness without acknowledgment"),
("Person_B","I said 'around' six — I didn't realize your life was governed by the precision of an atomic clock.","contempt",True,"sarcastic gaslighting about the time"),
("Person_A","Two hours is not 'around' anything — it is a completely different part of the evening.","anger",False,"challenging the unreasonable lateness"),
("Person_B","Your commitment to linear time is truly inspiring — I wish I lived in such a predictable world.","disapproval",True,"sarcastic disapproval of the other's expectation"),
("Person_A","I am not going to sit here and listen to you make me out to be the problem for expecting basic respect.","submission",False,"yielding to the reality of the toxic interaction"),
("Person_B","Oh, the 'basic respect' card — how original and not at all dramatic of you.","contempt",True,"sarcastic contempt for the other's feelings"),
]},
{"scenario":"casual","topic":"sarcasm_weather","utterances":[
("Person_A","It has been raining for twelve consecutive days and the basement is starting to smell like a swamp.","sadness",False,"prolonged bad weather causing property issues"),
("Person_B","I love this for us — I've always wanted to live in a coastal wetland without the hassle of moving.","joy",True,"sarcastic joy about the flooding"),
("Person_A","The roof is leaking in the guest room and the local hardware store is completely out of tarps.","fear",False,"escalating damage and lack of supplies"),
("Person_B","Don't worry — I'm sure the rain will stop the moment the entire house is fully submerged.","optimism",True,"sarcastic optimism about the situation"),
("Person_A","I am seriously considering buying an inflatable raft just to get to the mailbox tomorrow.","annoyance",False,"frustration with extreme weather"),
("Person_B","Make sure it's a stylish one — if we are going to be climate refugees, we should look our best.","contempt",True,"sarcastic contempt for the situation"),
]},
{"scenario":"workplace","topic":"sarcasm_corporate_speak","utterances":[
("CEO","We are initiating a strategic pivot to optimize our synergy and leverage our core competencies.","trust",False,"generic corporate jargon used"),
("Developer","Translation: We are firing the entire QA team and hoping the customers don't notice the bugs.","contempt",True,"sarcastic translation of corporate speak"),
("CEO","This is a lean-forward moment for the organization to embrace a more agile and resilient mindset.","vigilance",False,"more jargon to mask layoffs"),
("Developer","I'm leaning forward so much I might actually fall directly into a different company's recruitment pipeline.","aggressiveness",True,"sarcastic aggressive threat to leave"),
("CEO","We value your transparency and we are all in this boat together as one unified team.","love",False,"hollow corporate sentiment"),
("Developer","I hope the boat has enough life jackets for the people you just pushed overboard.","disapproval",True,"sarcastic disapproval of the layoffs"),
]},
{"scenario":"friendship","topic":"sarcasm_advice","utterances":[
("Friend_1","I'm thinking about getting back together with my ex who blocked me on everything last month.","apprehension",False,"considering a bad relationship decision"),
("Friend_2","What a brilliant idea — because the first three times you broke up were just practice rounds.","contempt",True,"sarcastic mockery of the decision"),
("Friend_1","He said he's changed and that he's been doing a lot of 'inner work' recently.","trust",False,"believing a likely lie"),
("Friend_2","I'm sure his 'inner work' is as deep and meaningful as a puddle in a parking lot.","disapproval",True,"sarcastic disapproval of the ex's claims"),
("Friend_1","You don't have to be so cynical — people can actually change if they want to.","sadness",False,"hurt by the lack of support"),
("Friend_2","I'm not being cynical, I'm being a fan of reality — but please, do proceed with the catastrophe.","submission",True,"sarcastic submission to the friend's choice"),
]},
{"scenario":"family","topic":"sarcasm_parenting","utterances":[
("Parent_1","The toddler managed to paint the entire living room wall with a permanent marker in five minutes.","rage",False,"child's destructive behavior"),
("Parent_2","I've always said this room lacked a certain 'post-modern chaotic' energy — he's a genius.","joy",True,"sarcastic joy at the mess"),
("Parent_1","He specifically chose the wall that we just repainted last weekend for his 'masterpiece'.","grief",False,"unfortunate timing of the mess"),
("Parent_2","His artistic instincts are impeccable — he clearly understands the value of a fresh canvas.","admiration",True,"sarcastic admiration of the child's 'choice'"),
("Parent_1","I am going to spend my entire Saturday scrubbing walls instead of actually relaxing.","sadness",False,"loss of personal time"),
("Parent_2","Think of it as a bonding exercise with the drywall — I'm sure it will be deeply fulfilling.","optimism",True,"sarcastic optimism about the cleaning task"),
]},
{"scenario":"workplace","topic":"sarcasm_deadlines","utterances":[
("Project_Mgr","I need the full quarterly report finished by five today even though I only gave you the data at noon.","vigilance",False,"unreasonable deadline imposed"),
("Analyst","Oh, certainly — I'll just use my time-turner to squeeze eight hours of work into the next five minutes.","contempt",True,"sarcastic response to unreasonable deadline"),
("Project_Mgr","The client is expecting a high-level executive summary with detailed visualizations of every metric.","trust",False,"manager adding more requirements"),
("Analyst","I'll make sure to include some interactive 3D models while I'm ignoring the laws of physics.","disapproval",True,"sarcastic disapproval of the scope creep"),
("Project_Mgr","I knew I could count on you to go above and beyond for the team on this one.","love",False,"manager using 'team' sentiment to manipulate"),
("Analyst","My dedication to your poor planning is truly the cornerstone of this entire department.","remorse",True,"sarcastic remorse about being a part of it"),
]},
{"scenario":"conflict","topic":"sarcasm_apology","utterances":[
("Person_A","I'm sorry I forgot your birthday — I've just been so busy with work that it completely slipped my mind.","remorse",False,"forgotten birthday apology"),
("Person_B","Don't worry, I barely noticed — it's not like birthdays happen on the same day every single year.","rage",True,"sarcastic rage about the forgotten birthday"),
("Person_A","I'll make it up to you this weekend with a nice dinner at that place you like.","trust",False,"attempt to make amends"),
("Person_B","A dinner? Wow, you're really pulling out all the stops to compensate for thirty years of friendship.","contempt",True,"sarcastic contempt for the gesture"),
("Person_A","I know I messed up and I'm genuinely trying to do something nice to fix it.","submission",False,"yielding to the criticism"),
("Person_B","Your 'genuine effort' is truly touching — I'll be sure to mark it on my calendar for next year.","disapproval",True,"sarcastic disapproval of the effort"),
]},
{"scenario":"casual","topic":"sarcasm_technology","utterances":[
("Person_A","My new laptop just decided to install a four-hour update right in the middle of my presentation.","rage",False,"untimely software update"),
("Person_B","I love how modern technology anticipates our needs — it clearly knew you needed a forced break.","joy",True,"sarcastic joy at the technological failure"),
("Person_A","I lost forty minutes of unsaved work and the client just stared at a spinning wheel the whole time.","grief",False,"loss of work and professional embarrassment"),
("Person_B","A spinning wheel is a very meditative image — I'm sure the client felt very zen about the delay.","optimism",True,"sarcastic optimism about the client's reaction"),
("Person_A","I am going to throw this machine out of a very high window if it happens one more time.","aggressiveness",False,"aggressive threat toward the laptop"),
("Person_B","Make sure to record it — the 'unboxing' videos are old, but 'defenestration' videos are the future.","contempt",True,"sarcastic contempt for the situation"),
]},
{"scenario":"workplace","topic":"sarcasm_feedback","utterances":[
("Manager","Your performance review says you are 'meeting expectations' but we'd like to see more 'passion'.","disapproval",False,"vague and unhelpful performance feedback"),
("Employee","I'll start weeping with joy every time I open a spreadsheet — will that demonstrate enough passion?","contempt",True,"sarcastic response to 'passion' requirement"),
("Manager","We want you to feel a deep personal connection to our mission of providing enterprise cloud solutions.","love",False,"manager using emotional language for corporate goals"),
("Employee","My heart beats for scalable infrastructure — I actually have a tattoo of our logo on my left ventricle.","submission",True,"sarcastic submission to the 'passion' requirement"),
("Manager","I'm glad we are on the same page — let's target a fifteen percent increase in passion for next quarter.","trust",False,"manager taking the sarcasm literally"),
("Employee","I'll reach out to my spiritual advisor to see if we can optimize my soul for better throughput.","remorse",True,"sarcastic remorse about soul optimization"),
]},
{"scenario":"conflict","topic":"sarcasm_boundaries","utterances":[
("Person_A","I took the liberty of going through your mail while you were away just to make sure nothing was urgent.","vigilance",False,"invasion of privacy defended as helpfulness"),
("Person_B","How incredibly thoughtful of you — I've always wanted a personal secretary who doesn't respect boundaries.","rage",True,"sarcastic rage at the invasion of privacy"),
("Person_A","I was just trying to be helpful because I know how disorganized you can be with paperwork.","contempt",False,"insulting the other person while defending the action"),
("Person_B","Your 'helpfulness' is as welcome as a swarm of bees in a small elevator.","disapproval",True,"sarcastic disapproval of the 'help'"),
("Person_A","Fine, I won't bother helping you with anything ever again since you are so ungrateful.","anger",False,"defensive anger from the intruder"),
("Person_B","Is that a promise? Because that would be the best gift you've ever given me.","joy",True,"sarcastic joy at the prospect of being left alone"),
]},
{"scenario":"social","topic":"sarcasm_etiquette","utterances":[
("Guest_1","She arrived two hours late to the wedding and then complained that they ran out of the salmon.","disgust",False,"witnessing entitled behaviour"),
("Guest_2","Her timing is impeccable — she managed to miss the ceremony but arrived just in time for the critique.","contempt",True,"sarcastic mockery of the late guest"),
("Guest_1","She actually asked the bride if she could have a piece of the cake before the formal cutting.","disapproval",False,"disapproval of guest's rudeness"),
("Guest_2","A true paragon of social etiquette — we should all aspire to her level of unearned confidence.","contempt",True,"sarcastic contempt for the guest's behavior"),
("Guest_1","I am going to start charging her a 'rudeness tax' every time she speaks to me today.","aggressiveness",False,"aggressive plan to deal with the guest"),
("Guest_2","You'll be a billionaire by the end of the cocktail hour — don't forget the little people.","joy",True,"sarcastic joy at the 'tax' plan"),
]},
{"scenario":"workplace","topic":"sarcasm_innovation_work","utterances":[
("Director","We've decided to replace the coffee machine with a 'mindfulness station' to improve employee wellness.","joy",False,"unpopular change framed as a benefit"),
("Engineer","Perfect — because nothing solves a 3 AM production outage quite like a three-minute breathing exercise.","contempt",True,"sarcastic response to the coffee machine removal"),
("Director","We believe that inner peace is more sustainable for long-term productivity than caffeine.","trust",False,"director justifying the change"),
("Engineer","I'll tell the database to remain calm while it's corrupting the entire customer table.","disapproval",True,"sarcastic disapproval of the 'wellness' focus"),
("Director","Your skepticism is just part of the transition process — you'll embrace the silence soon enough.","vigilance",False,"director dismissing the engineer's concern"),
("Engineer","I'm embracing the silence of my career prospects at this company as we speak.","submission",True,"sarcastic submission to the new policy"),
]},
{"scenario":"conflict","topic":"sarcasm_broken_promises","utterances":[
("Person_A","I know I said I'd help you move today but something came up and I'm going to the beach instead.","remorse",False,"breaking a significant promise for a trivial reason"),
("Person_B","Oh, please don't let my entire life packed into boxes get in the way of your tan.","rage",True,"sarcastic rage at the broken promise"),
("Person_A","I'll come over tomorrow morning and help you with the heavy stuff for a couple of hours.","trust",False,"weak attempt to offer future help"),
("Person_B","Tomorrow morning? That's so generous — I'm sure the movers will love waiting around for you.","contempt",True,"sarcastic contempt for the offer"),
("Person_A","I'm just one person, it's not like my presence would have changed that much anyway.","submission",False,"minimizing their own importance to deflect guilt"),
("Person_B","Your self-awareness is truly stunning — it's almost as impressive as your reliability.","disapproval",True,"sarcastic disapproval of the person's character"),
]},
{"scenario":"casual","topic":"sarcasm_bad_luck","utterances":[
("Person_A","I just dropped my phone in the toilet, and when I reached for it, I knocked my glasses in too.","grief",False,"series of unfortunate events"),
("Person_B","Well, at least now your phone and your glasses can get to know each other in a new environment.","joy",True,"sarcastic joy at the double accident"),
("Person_A","I don't have insurance on either of them and I'm pretty sure the water damage is terminal.","fear",False,"concern about the cost and loss"),
("Person_B","On the bright side, you've successfully avoided all those annoying notifications for the rest of the day.","optimism",True,"sarcastic optimism about the situation"),
("Person_A","I can't even see the screen to check if it's working because my glasses are also soaking wet.","annoyance",False,"frustration with the situation"),
("Person_B","A truly immersive sensory experience — you should write a blog post about it, once you can see again.","contempt",True,"sarcastic contempt for the bad luck"),
]},
{"scenario":"workplace","topic":"sarcasm_bureaucracy","utterances":[
("Employee","I need a signature for this ten-dollar expense and I've been told I need four different approvals.","annoyance",False,"excessive bureaucracy for a small task"),
("Manager","The new policy ensures maximum fiscal responsibility across all departments without exception.","trust",False,"manager defending the bureaucratic policy"),
("Employee","I've spent fifty dollars of my hourly rate just trying to get this ten dollars approved.","rage",False,"pointing out the inefficiency"),
("Manager","The process is the product — we value the integrity of the system over mere efficiency.","vigilance",False,"manager prioritizing the system"),
("Employee","I'll make sure to document the 'integrity' of this process in my next resignation letter.","aggressiveness",True,"sarcastic aggressive threat to quit"),
("Manager","Your feedback has been noted and will be reviewed by the process improvement committee.","acceptance",False,None),
("Employee","I can't wait for the committee to meet in six months to decide that more forms are the answer.","contempt",True,"sarcastic contempt for the committee"),
]},
{"scenario":"workplace","topic":"sarcasm_meeting_length","utterances":[
("Employee_A","That meeting was supposed to be thirty minutes but we are currently at hour three.","boredom",False,"excessive meeting duration"),
("Employee_B","I'm enjoying the gradual entropy of my will to live — it's a very thorough process.","contempt",True,"sarcastic contempt for the meeting"),
("Employee_A","The slides are still on the first section and there are nine more to go.","grief",False,"slow progress of meeting"),
("Employee_B","I've started naming the dust motes in the projector beam — 'Dave' is very energetic today.","pensiveness",True,"sarcastic pensiveness about the meeting"),
("Employee_A","I think my legs have actually fused to this chair — I may need surgery to leave the room.","annoyance",True,"sarcastic annoyance at being trapped"),
("Employee_B","I'll write you a recommendation for a very comfortable wheelchair — for the next meeting.","optimism",True,"sarcastic optimism about the next meeting"),
]},
{"scenario":"social","topic":"sarcasm_fashion","utterances":[
("Guest","She spent four thousand dollars on a dress that looks like it was made from recycled car mats.","disgust",False,"judging expensive but ugly fashion"),
("Friend","It's 'avant-garde' — you clearly don't understand the complex dialogue between rubber and silk.","contempt",True,"sarcastic defense of the dress"),
("Guest","The dialogue seems to be mostly about how much money she has to throw away.","disapproval",False,"disapproval of wastefulness"),
("Friend","It's a statement on the industrial-military-catwalk complex — or she was just blindfolded while shopping.","contempt",True,"sarcastic mockery of the fashion statement"),
("Guest","I'm going to start wearing my floor mats and see if I get invited to the Met Gala.","aggressiveness",True,"sarcastic aggressive fashion plan"),
("Friend","I'll be your stylist — we can call the collection 'Toyota Corolla Summer'.","joy",True,"sarcastic joy at the fashion plan"),
]},
{"scenario":"romance","topic":"sarcasm_dating_apps","utterances":[
("Single_A","I just went on my fifth date this month where the person didn't look anything like their photos.","sadness",False,"disappointment with dating app deception"),
("Single_B","Maybe they just have a very powerful imagination and a very old camera.","optimism",True,"sarcastic optimism about the deception"),
("Single_A","One guy used a photo from his high school graduation — he is currently forty-two.","rage",False,"extreme age deception on dating app"),
("Single_B","He's a time traveler — you should be honored to witness such a collapse of the space-time continuum.","amazement",True,"sarcastic amazement at the deception"),
("Single_A","I'm deleting the app and moving to a remote cabin where my only companion will be a bear.","submission",False,"giving up on modern dating"),
("Single_B","The bear will at least be honest about its intention to eat you — which is a step up.","contempt",True,"sarcastic contempt for dating apps"),
]},
{"scenario":"workplace","topic":"sarcasm_it_support","utterances":[
("User","My computer won't turn on and IT told me to 'try breathing deeply' before they send a technician.","rage",False,"unhelpful IT support response"),
("Colleague","They're moving into 'holistic hardware support' — the motherboard just needs some positive energy.","contempt",True,"sarcastic mockery of IT response"),
("User","I have a deadline in two hours and I'm currently performing a 'tech-exorcism' with a paperclip.","fear",False,"desperation due to technical failure"),
("Colleague","I'm sure the paperclip will appreciate the opportunity to lead such a high-stakes project.","optimism",True,"sarcastic optimism about the paperclip"),
("User","I'm going to walk over to the IT office and 'breathe deeply' directly into their server room.","aggressiveness",False,"aggressive intent toward IT"),
("Colleague","Record the screaming — it will be a great addition to the 'corporate wellness' seminar.","joy",True,"sarcastic joy at the conflict"),
]},
{"scenario":"conflict","topic":"sarcasm_bad_advice","utterances":[
("Person_A","I told him I was feeling overwhelmed and he suggested I 'just try being less busy'.","disapproval",False,"receiving dismissive and obvious advice"),
("Person_B","A visionary! I wonder why the rest of us haven't discovered the 'do less work' strategy.","contempt",True,"sarcastic mockery of the advice"),
("Person_A","He then offered to lend me his book on 'The Art of Doing Nothing' while he went on vacation.","anger",False,"further insult from the advice-giver"),
("Person_B","The irony is so thick you could use it to insulate a house — or at least a very small room.","pensiveness",True,"sarcastic pensiveness about the irony"),
("Person_A","I'm going to give him my own book: 'The Art of Finding Your Own Way Home' — from the woods.","aggressiveness",True,"sarcastic aggressive response"),
("Person_B","Make sure the maps are 'minimalist' — to reflect his intellectual depth.","contempt",True,"sarcastic contempt for the advisor"),
]},
{"scenario":"casual","topic":"sarcasm_public_transport","utterances":[
("Commuter_A","The train is forty minutes late and the announcement says it's due to 'unforeseen leaves'.","annoyance",False,"weak excuse for train delay"),
("Commuter_B","Those leaves are very sneaky — they've only been falling for three months, it's a total surprise.","contempt",True,"sarcastic contempt for the excuse"),
("Commuter_A","There are currently six hundred people on the platform and only one functioning ticket machine.","rage",False,"poor station management"),
("Commuter_B","It's an 'efficiency test' — they want to see how many people can achieve enlightenment while waiting.","optimism",True,"sarcastic optimism about the wait"),
("Commuter_A","I think I'm going to achieve 'arrested for property damage' if the next train doesn't arrive.","aggressiveness",False,"aggressive frustration"),
("Commuter_B","I'll be your character witness — 'He was a quiet man, until the leaves happened.'","joy",True,"sarcastic joy at the situation"),
]},
{"scenario":"workplace","topic":"sarcasm_overtime","utterances":[
("Employee","The boss asked me to work this weekend because the project is 'behind' — despite my finishing all my tasks.","rage",False,"unjustified overtime request"),
("Colleague","Your reward for being efficient is the privilege of doing the work of the three people who aren't.","contempt",True,"sarcastic mockery of corporate reward system"),
("Employee","He said it would be a 'great learning experience' for me to handle the overflow.","disapproval",False,"manager's hollow justification"),
("Colleague","I'm sure you'll learn a lot about the 'physics of burnout' and the 'geometry of resentment'.","pensiveness",True,"sarcastic pensiveness about the consequences"),
("Employee","I'm going to 'learn' how to update my resume while sitting at my desk on Saturday.","aggressiveness",True,"sarcastic aggressive response to overtime"),
("Colleague","That's the spirit! Use the company's electricity to find a company that actually has some.","optimism",True,"sarcastic optimism about the job hunt"),
]},
{"scenario":"friendship","topic":"sarcasm_bad_cooking","utterances":[
("Friend_A","She served us a 'vegan lasagna' that was basically three layers of wet cardboard and a prayer.","disgust",False,"bad meal experience"),
("Friend_B","I liked the 'authentic industrial' texture — it really made me appreciate the value of teeth.","contempt",True,"sarcastic mockery of the food"),
("Friend_A","She asked for 'honest feedback' and I told her the color of the sauce was 'brave'.","disapproval",False,"giving polite but critical feedback"),
("Friend_B","'Brave' is the perfect word for a sauce that looks like it's planning a hostile takeover.","pensiveness",True,"sarcastic pensiveness about the sauce"),
("Friend_A","I'm bringing a hidden sandwich the next time we go there — for my own survival.","aggressiveness",False,"planned survival strategy"),
("Friend_B","Bring two — I'll pay you in actual food once we escape the 'culinary innovation' zone.","joy",True,"sarcastic joy at the escape plan"),
]},
{"scenario":"workplace","topic":"sarcasm_office_plants","utterances":[
("Admin","We've spent two thousand dollars on 'living walls' while the coffee machine still leaks brown water.","disapproval",False,"misallocated office budget"),
("Developer","The plants need a high-end environment to thrive — unlike the employees, who are basically moss.","contempt",True,"sarcastic comparison of plants and employees"),
("Admin","One of the plants died within three days and they're calling in an 'organic systems consultant'.","rage",False,"absurd response to plant death"),
("Developer","I'm sure the consultant will have a very deep conversation with the ferns about their 'career goals'.","optimism",True,"sarcastic optimism about the consultant"),
("Admin","I'm going to start 'consulting' with the dead plant — it's a better listener than my manager.","submission",True,"sarcastic submission to the absurdity"),
("Developer","The plant has more growth potential too — at least it's honest about being compost.","contempt",True,"sarcastic contempt for management"),
]},
{"scenario":"social","topic":"sarcasm_gym_culture","utterances":[
("Gym_Goer","There's a guy who spends forty minutes occupying the only squat rack just to take selfies.","annoyance",False,"frustrating gym behaviour"),
("Friend","He's building his 'digital core' — physical strength is so last decade, it's all about the lighting.","contempt",True,"sarcastic mockery of gym selfies"),
("Gym_Goer","He actually asked me to move so he could get the 'ideal angle' of his water bottle.","rage",False,"extreme entitlement at the gym"),
("Friend","A true artist at work — we should be honored to be background extras in his fitness odyssey.","amazement",True,"sarcastic amazement at the entitlement"),
("Gym_Goer","I'm going to 'accidentally' drop a plate near his phone and see if it improves the composition.","aggressiveness",False,"aggressive intent toward the narcissist"),
("Friend","Do it! I'll provide the 'candid reaction' shot for his inevitable apology video.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_buzzwords","utterances":[
("Manager","We need to circle back and deep-dive into our synergistic deliverables to ensure we are leaning in.","boredom",False,"excessive use of corporate buzzwords"),
("Employee","I'm leaning in so far I've actually achieved a horizontal position — it's very efficient for napping.","contempt",True,"sarcastic response to buzzwords"),
("Manager","We're looking for a 'culture-add' who can 'disrupt' our 'legacy paradigms' with 'blue-sky thinking'.","trust",False,"more jargon to describe a simple job"),
("Employee","I'll bring my 'cloud-native' enthusiasm and my 'serverless' work ethic to the paradigm shift.","submission",True,"sarcastic submission to the jargon"),
("Manager","That's the 'growth mindset' we need — let's socialize this with the key stakeholders.","optimism",False,"manager taking the sarcasm as a positive"),
("Employee","I'll 'socialize' it until it's a fully integrated member of the corporate country club.","disapproval",True,"sarcastic disapproval of the process"),
]},
{"scenario":"casual","topic":"sarcasm_internet_speed","utterances":[
("User","My internet is so slow that I just watched a 'loading' circle for forty-five minutes.","annoyance",False,"frustration with slow internet"),
("Friend","It's a 'digital patience exercise' — your ISP is just trying to help you achieve a state of Zen.","optimism",True,"sarcastic optimism about the slow speed"),
("User","I'm paying for 'ultra-fiber' and currently I'm getting 'single-thread-of-silk' speeds.","rage",False,"paying for service not received"),
("Friend","Ultra-fiber is a relative term — relative to a carrier pigeon, you're doing great.","contempt",True,"sarcastic contempt for the ISP"),
("User","I'm going to send my complaint via a carrier pigeon — it'll probably get there faster than this email.","aggressiveness",True,"sarcastic aggressive response to slow tech"),
("Friend","Make sure the pigeon has 'low latency' — we don't want any 'feather-loss' in transit.","joy",True,"sarcastic joy at the pigeon plan"),
]},
{"scenario":"social","topic":"sarcasm_wedding_costs","utterances":[
("Guest_A","They're charging two hundred dollars a plate and the main course was a single artisanal pea.","disgust",False,"expensive but tiny wedding meal"),
("Guest_B","The pea had a very 'curated' life — it probably went to a better school than we did.","contempt",True,"sarcastic mockery of the expensive food"),
("Guest_A","There was a 'monogrammed water station' and a 'bespoke napkin consultant' on the invitation.","disapproval",False,"excessive and pretentious wedding details"),
("Guest_B","I'm sure the napkin consultant has a PhD in 'foldable aesthetics' — it's a very rigorous field.","pensiveness",True,"sarcastic pensiveness about the profession"),
("Guest_A","I'm going to start 'consulting' on the proper way to eat a two-hundred-dollar pea — very slowly.","aggressiveness",True,"sarcastic aggressive plan"),
("Guest_B","I'll be your 'consumption strategist' — we can charge by the calorie.","joy",True,"sarcastic joy at the business idea"),
]},
{"scenario":"workplace","topic":"sarcasm_remote_work","utterances":[
("Manager","We've decided to move to a 'mandatory hybrid' model where everyone has to be in the office on Tuesdays.","annoyance",False,"unpopular change to remote work policy"),
("Employee","Oh, brilliant — because my productivity really peaks when I'm surrounded by people having loud lunches.","contempt",True,"sarcastic response to office return"),
("Manager","It's for 'spontaneous collaboration' and to 'rebuild the social fabric' of the company.","trust",False,"manager's cliché justification"),
("Employee","I've already 'collaborated' with the elevator — it told me that the 'social fabric' is mostly polyester.","disapproval",True,"sarcastic disapproval of the justification"),
("Manager","I appreciate your 'creative engagement' with the new policy — see you Tuesday.","acceptance",False,None),
("Employee","I'll bring my 'fabric softener' — to help with all that spontaneous social rebuilding.","submission",True,"sarcastic submission to the policy"),
]},
{"scenario":"casual","topic":"sarcasm_celebrity_news","utterances":[
("Person_A","I just read a three-page article about a celebrity who 'discovered' that water is important for hydration.","boredom",False,"trivial celebrity news"),
("Person_B","What a breakthrough! I wonder if they'll win the Nobel Prize for 'Groundbreaking Liquid Awareness'.","contempt",True,"sarcastic mockery of the news"),
("Person_A","The article had a 'step-by-step guide' on how to hold a glass of water 'fashionably'.","disgust",False,"absurdity of lifestyle journalism"),
("Person_B","I've been holding mine 'unfashionably' for years — my social standing is currently in the gutter.","sadness",True,"sarcastic sadness about their own 'failure'"),
("Person_A","I'm going to write a book about the 'existential significance of blinking' — and sell it for fifty dollars.","aggressiveness",True,"sarcastic aggressive business plan"),
("Person_B","I'll buy the first ten copies — I need to know if I'm blinking with enough 'intention'.","optimism",True,"sarcastic optimism about the blinking book"),
]},
{"scenario":"workplace","topic":"sarcasm_it_security","utterances":[
("User","IT just made me change my password for the fourth time this month — it now needs to be twenty characters long.","annoyance",False,"excessive security requirements"),
("Colleague","They want to make sure your password is more secure than the actual server it's protecting.","contempt",True,"sarcastic mockery of security policy"),
("User","My current password is a combination of my first pet's name and the GPS coordinates of my high school.","pensiveness",False,"complexity of current password"),
("Colleague","Very secure — as long as no one knows you went to high school or ever had a dog.","optimism",True,"sarcastic optimism about the security"),
("User","I'm going to start writing it on a giant neon sign and putting it on my desk for 'convenience'.","aggressiveness",True,"sarcastic aggressive response to security"),
("Colleague","I'll help you with the wiring — we can call it 'transparent authentication'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_wine_tasting","utterances":[
("Guest_A","The sommelier said this wine has 'notes of damp earth' and a 'whisper of forgotten dreams'.","boredom",False,"pretentious wine descriptions"),
("Guest_B","I'm mostly getting 'notes of sour grapes' and a 'loud scream of twenty dollars wasted'.","contempt",True,"sarcastic response to the wine"),
("Guest_A","He actually asked me to 'inhale the essence of the terroir' before I took a sip.","disgust",False,"absurdity of the tasting ritual"),
("Guest_B","The 'essence of the terroir' smells suspiciously like the dirt in my backyard after a storm.","disapproval",True,"sarcastic disapproval of the wine's quality"),
("Guest_A","I'm going to start 'essentializing' my tap water — it has 'notes of pipe' and 'hints of chlorine'.","aggressiveness",True,"sarcastic aggressive plan"),
("Guest_B","I'll be your 'water sommelier' — we can charge people to 'experience the hydration'.","optimism",True,"sarcastic optimism about the business"),
]},
{"scenario":"workplace","topic":"sarcasm_mentorship","utterances":[
("Junior","My mentor told me that the key to success is to 'always be the last person to leave the office'.","annoyance",False,"bad career advice based on overwork"),
("Senior","He's right — the cleaners really appreciate having someone to talk to while they mop the floors.","contempt",True,"sarcastic mockery of the advice"),
("Junior","He also said that 'sleep is a weakness that can be overcome with enough caffeine and ambition'.","fear",False,"toxic productivity culture"),
("Senior","I'm sure your 'ambition' will look great on your medical chart when you collapse from exhaustion.","pensiveness",True,"sarcastic pensiveness about the consequences"),
("Junior","I'm going to 'mentor' him on the 'art of seeing his family' — by mailing him a photograph of them.","aggressiveness",True,"sarcastic aggressive response"),
("Senior","Make sure it's a high-resolution photo — so he can remember what they look like in 3D.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_food_delivery","utterances":[
("User","My food delivery is two hours late and the app says the driver is currently 'finding his purpose'.","annoyance",False,"extreme delay and weird status message"),
("Friend","He's probably at a retreat — the burrito is just part of his journey to spiritual enlightenment.","optimism",True,"sarcastic optimism about the driver's delay"),
("User","I'm so hungry I'm starting to consider if the napkins are actually edible if I add enough salt.","grief",False,"extreme hunger due to delay"),
("Friend","Napkins are 'zero-calorie fiber' — you're basically on a very advanced detox program.","contempt",True,"sarcastic contempt for the situation"),
("User","I'm going to 'find my purpose' by walking to the restaurant and 'manifesting' my own dinner.","aggressiveness",True,"sarcastic aggressive response to delivery failure"),
("Friend","I'll be your 'manifestation coach' — for a small fee of one half of your burrito.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_home_decor","utterances":[
("Guest","She decorated her entire house in 'minimalist beige' — even the cat is currently off-white.","disgust",False,"boring and monochrome home decor"),
("Friend","It's 'Scandinavian serenity' — color is far too emotional and distracting for a truly modern home.","contempt",True,"sarcastic defense of the decor"),
("Guest","The living room looks like a high-end dentist's waiting room — I'm waiting for someone to call my name.","disapproval",False,"uncomfortable and sterile environment"),
("Friend","I'm sure the 'serenity' is very helpful for when you're staring at the beige wall for four hours.","pensiveness",True,"sarcastic pensiveness about the decor"),
("Guest","I'm going to 'disrupt' her serenity by bringing a very bright red pillow the next time I visit.","aggressiveness",True,"sarcastic aggressive plan"),
("Friend","A terrorist attack of color! I'll provide the 'chromatic cover' — I'll wear a yellow shirt.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_reorg","utterances":[
("Employee","We're having our third 'organizational restructuring' this year to 'optimize our vertical integration'.","annoyance",False,"frequent and disruptive company reorganizations"),
("Colleague","They're moving us from 'vertical' to 'horizontal' — soon we'll be 'diagonal' and then 'circular'.","contempt",True,"sarcastic mockery of reorg terminology"),
("Employee","My new manager is currently in a different time zone and doesn't know what my job actually is.","rage",False,"poor management resulting from reorg"),
("Colleague","A 'decentralized leadership model' — it's very innovative to have a boss who is basically a ghost.","optimism",True,"sarcastic optimism about the ghost boss"),
("Employee","I'm going to 'restructure' my own attendance — by being 'integrated' into my own couch tomorrow.","submission",True,"sarcastic submission to the chaos"),
("Colleague","I'll be your 'couch consultant' — we can 'optimize' our 'leisure-time deliverables' together.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_movie_reboots","utterances":[
("Movie_Fan","They just announced a reboot of a movie that came out only three years ago.","boredom",False,"unnecessary movie reboot"),
("Friend","The 'creative well' is so deep they've started digging through the bottom into the basement.","contempt",True,"sarcastic mockery of Hollywood's lack of ideas"),
("Movie_Fan","The new version is exactly the same, but now it's 'grittier' and everyone is 'dark and brooding'.","disgust",False,"clichéd reboot strategy"),
("Friend","I'm sure the 'grittiness' will distract us from the fact that we've already seen this exact story.","pensiveness",True,"sarcastic pensiveness about the reboot"),
("Movie_Fan","I'm going to reboot my own childhood — starting with a 'dark and brooding' version of my fifth birthday.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll play the 'gritty' clown — I'll make the balloon animals out of barbed wire.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_training","utterances":[
("New_Hire","The 'onboarding training' is a six-hour video of the CEO talking about his 'journey' to the top.","boredom",False,"unhelpful and self-indulgent training"),
("Mentor","It's an 'inspirational deep-dive' — you're supposed to absorb his 'essence' through the screen.","contempt",True,"sarcastic mockery of the training video"),
("New_Hire","There was a 'mandatory quiz' at the end where I had to name his favorite childhood pet.","disapproval",False,"irrelevant and absurd training content"),
("Mentor","His pet's name is 'Success' — I'm sure the irony is purely intentional and not at all accidental.","pensiveness",True,"sarcastic pensiveness about the pet's name"),
("New_Hire","I'm going to 'onboard' my own 'journey' — by walking directly out of the front door.","aggressiveness",True,"sarcastic aggressive response to training"),
("Mentor","I'll be your 'travel agent' — I can recommend a very 'inspirational' coffee shop across the street.","optimism",True,"sarcastic optimism about the escape"),
]},
{"scenario":"social","topic":"sarcasm_pet_owners","utterances":[
("Neighbor","She spent five hundred dollars on a 'psychic' for her dog because he 'seemed a bit distant'.","disgust",False,"excessive and absurd spending on pets"),
("Friend","The dog is probably just 're-evaluating his relationship with his tail' — it's a very complex bond.","contempt",True,"sarcastic mockery of the pet psychic"),
("Neighbor","The psychic said the dog was a 'French poet in a past life' and needs more 'intellectual stimulation'.","amazement",True,"sarcastic amazement at the psychic's claims"),
("Friend","I'll lend him my copy of 'The Stranger' — I'm sure he'll find the 'absurdity' very relatable.","joy",True,"sarcastic joy at the idea of a dog reading Camus"),
("Neighbor","I'm going to start 'consulting' with my cat — I'm pretty sure she thinks she's a 'disappointed goddess'.","submission",True,"sarcastic submission to the absurdity"),
("Friend","She doesn't think she is — she *is* a disappointed goddess, and you're the unworthy servant.","admiration",True,"sarcastic admiration of the cat's attitude"),
]},
{"scenario":"workplace","topic":"sarcasm_open_office","utterances":[
("Employee","They've removed the cubicle walls to 'foster a sense of community' and now I can hear my neighbor's chewing.","annoyance",False,"negative impact of open office plans"),
("Colleague","It's a 'sensory-rich environment' — you're supposed to 'integrate' the sound of chips into your workflow.","contempt",True,"sarcastic response to the noise"),
("Employee","I've achieved 'maximum community' — I now know more about my coworker's weekend than I do about my own job.","disapproval",False,"lack of privacy in open office"),
("Colleague","I'm sure the 'spontaneous collaboration' will start the moment you both run out of chips.","optimism",True,"sarcastic optimism about the collaboration"),
("Employee","I'm going to 'collaborate' on a new project: 'The Geometry of Soundproof Headphones'.","aggressiveness",True,"sarcastic aggressive response to the noise"),
("Colleague","I'll be your 'acoustic engineer' — we can build a 'fortress of solitude' out of empty boxes.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_self_checkout","utterances":[
("Shopper","The self-checkout machine just told me there's an 'unexpected item in the bagging area' for the tenth time.","annoyance",False,"frustrating experience with self-checkout"),
("Friend","The machine is just 'emotionally attached' to your groceries — it doesn't want to let them go.","optimism",True,"sarcastic optimism about the technical error"),
("Shopper","I'm currently 'fighting' a robot over a bag of kale and I'm pretty sure the robot is winning.","submission",True,"sarcastic submission to the technology"),
("Friend","The kale is a 'high-stakes asset' — the machine is just 'protecting the inventory' with its life.","contempt",True,"sarcastic contempt for the machine's logic"),
("Shopper","I'm going to 'bag' the machine and see if it appreciates the 'unexpected item' in its own area.","aggressiveness",True,"sarcastic aggressive response to the machine"),
("Friend","I'll provide the 'technical support' — I'll stand there and look 'thoughtful' while you do it.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_wellness_apps","utterances":[
("Employee","The company just gave us a 'wellness app' that sends us reminders to 'breathe' every fifteen minutes.","boredom",False,"unhelpful corporate wellness initiative"),
("Colleague","It's 'digital life support' — in case we forget how to perform basic biological functions during the workday.","contempt",True,"sarcastic mockery of the wellness app"),
("Employee","I just got a notification telling me to 'visualize a calm ocean' while I'm fixing a server crash.","rage",False,"bad timing of wellness notifications"),
("Colleague","The ocean is probably also 'crashing' — it's a very 'integrated' wellness experience.","pensiveness",True,"sarcastic pensiveness about the situation"),
("Employee","I'm going to 'visualize' the app's deletion — it's a very 'calming' mental exercise for me.","aggressiveness",True,"sarcastic aggressive response to the app"),
("Colleague","I'll be your 'mindfulness coach' — I'll tell you to 'ignore the notifications' with 'great intention'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_modern_art","utterances":[
("Guest","The centerpiece of this gallery is a single white chair in a completely empty white room.","boredom",False,"judging minimalist and pretentious art"),
("Friend","It's 'The Silence of Possibility' — the chair is a 'metaphor for the burden of existence'.","contempt",True,"sarcastic defense of the art"),
("Guest","I thought it was just a chair that the janitor forgot to move before the gallery opened.","disapproval",False,"lack of appreciation for the 'art'"),
("Friend","Your 'literalist perspective' is so quaint — you're missing the 'visceral dialogue' with the space.","pensiveness",True,"sarcastic pensiveness about the art's meaning"),
("Guest","I'm going to start a 'visceral dialogue' with the chair — by sitting on it and 'experiencing the comfort'.","aggressiveness",True,"sarcastic aggressive plan"),
("Friend","A performance art piece! We can call it 'The Hubris of the Seated Guest' — it'll be a sensation.","joy",True,"sarcastic joy at the idea"),
]},
{"scenario":"workplace","topic":"sarcasm_office_snack_room","utterances":[
("Employee","The snack room is currently 'stocked' with three boxes of gluten-free crackers and a bowl of very sad lemons.","annoyance",False,"poor selection of office snacks"),
("Colleague","It's a 'curated selection of minimalist nutrition' — they want us to 'focus on the crunch'.","contempt",True,"sarcastic response to the snacks"),
("Employee","I'm pretty sure the lemons have been there since the previous CEO left in 2018.","disgust",False,"unfresh office food"),
("Colleague","They're 'vintage' — it's a 'historical flavor profile' that's very difficult to achieve.","pensiveness",True,"sarcastic pensiveness about the lemons"),
("Employee","I'm going to 'innovate' a new snack: 'Lemon-infused Desperation' — it's very 'on-brand' for this month.","aggressiveness",True,"sarcastic aggressive response to the snacks"),
("Colleague","I'll be your 'marketing strategist' — we can target the 'nutritionally underserved' developer demographic.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_smart_homes","utterances":[
("Owner","My 'smart' lights just decided to turn the kitchen into a 'disco mode' while I was trying to cook dinner.","annoyance",False,"unreliable smart home technology"),
("Friend","The house is just 'celebrating your culinary journey' — it's a very supportive environment.","optimism",True,"sarcastic optimism about the tech failure"),
("Owner","I had to 're-authenticate' my toaster this morning because it 'lost its connection to the cloud'.","rage",False,"absurdity of internet-connected appliances"),
("Friend","A toaster without a cloud connection is just a 'bread-heating unit' — it's so 'legacy'.","contempt",True,"sarcastic contempt for basic appliances"),
("Owner","I'm going to 'de-smart' my house — by hitting the 'smart hub' with a very 'un-smart' hammer.","aggressiveness",True,"sarcastic aggressive response to smart tech"),
("Friend","I'll provide the 'analog support' — I'll bring the matches for the 'original smart lighting'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_home_renovation","utterances":[
("Owner","The contractor told me that the 'open-concept' kitchen will take six months longer because of 'unexpected walls'.","rage",False,"unreasonable delay in home renovation"),
("Neighbor","Those walls are very 'improvisational' — they probably just wanted to 'explore the space' before they left.","contempt",True,"sarcastic mockery of the contractor's excuse"),
("Owner","I've been living in a 'dust-centric' environment for so long I've forgotten what a floor looks like.","sadness",False,"long-term discomfort due to renovation"),
("Neighbor","Dust is just 'unorganized material' — you're basically a 'collector of potential floors'.","pensiveness",True,"sarcastic pensiveness about the situation"),
("Owner","I'm going to 'renovate' the contractor's face — with a very 'open-concept' fist.","aggressiveness",True,"sarcastic aggressive threat"),
("Neighbor","I'll be your 'legal advisor' — we can call it 'structural integrity testing'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_office_supplies","utterances":[
("Admin","We've replaced the high-quality pens with 'eco-friendly' sticks that stop working after two sentences.","annoyance",False,"cost-cutting on office supplies"),
("Developer","It's to 'encourage concise communication' — the company wants us to 'be brief or be silent'.","contempt",True,"sarcastic response to the bad pens"),
("Admin","I've had to 'borrow' pens from the bank across the street just to fill out the supply request form.","disapproval",False,"absurdity of the supply situation"),
("Developer","I'm sure the bank appreciates your 'unauthorized resource sharing' — it's very 'community-focused'.","optimism",True,"sarcastic optimism about the theft"),
("Admin","I'm going to start 'writing' my emails in pencil — and then 'sending' them via a very slow fax machine.","submission",True,"sarcastic submission to the poor supplies"),
("Developer","I'll be your 'analog technician' — I'll help you 'sharpen the communication' with a very sharp knife.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_streaming_services","utterances":[
("User","My streaming service just increased the price by five dollars and added a 'mandatory ad tier'.","annoyance",False,"price hike and worse service on streaming"),
("Friend","They want to 're-introduce you to the nostalgic charm of commercials' — it's a very 'vintage' experience.","contempt",True,"sarcastic response to the price hike"),
("User","I'm paying for 'ad-free' and I still have to watch a 'preview' of a show about people who make 'artisanal mud'.","disgust",False,"unwanted content on 'ad-free' service"),
("Friend","'Artisanal mud' is a very 'grounded' genre — you're just not the 'target demographic' for soil-based drama.","pensiveness",True,"sarcastic pensiveness about the content"),
("User","I'm going to 'cancel' my subscription — and then 'stream' my own life by staring out of the window.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'executive producer' — I'll tell you when the 'plot' of the street is getting too 'repetitive'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_party_etiquette","utterances":[
("Guest_A","He brought a single bottle of lukewarm water to a housewarming party and then ate half the cheese board.","disgust",False,"rude and cheap guest behaviour"),
("Guest_B","A true minimalist! He's 'reducing his carbon footprint' by 'utilizing other people's resources'.","contempt",True,"sarcastic mockery of the guest"),
("Guest_A","He also spent the whole night 'critiquing' the host's choice of flooring while standing on it.","disapproval",False,"rude guest behaviour"),
("Guest_B","I'm sure the floor was 'deeply moved' by his 'architectural insights' — it's a very 'empathetic' laminate.","pensiveness",True,"sarcastic pensiveness about the situation"),
("Guest_A","I'm going to 'housewarm' him — by 'warming' his seat with a very 'minimalist' amount of hot sauce.","aggressiveness",True,"sarcastic aggressive plan"),
("Guest_B","I'll provide the 'flavor enhancement' — I'll distract him with a very 'substantive' conversation about cheese.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_performance_plans","utterances":[
("Manager","We're putting you on a 'performance enhancement plan' to help you 'achieve your full potential'.","fear",False,"threat of job loss disguised as help"),
("Employee","Oh, I'm so excited — I've always wanted a 'plan' that involves more paperwork and less job security.","contempt",True,"sarcastic response to the PIP"),
("Manager","It's an 'opportunity for growth' and we want to see you 'blossom' in this new structure.","trust",False,"manager using positive language for a negative process"),
("Employee","I'll 'blossom' so much I'll probably 'pollinate' a different company by the end of the month.","aggressiveness",True,"sarcastic aggressive response to the PIP"),
("Manager","Your 'confidence' is noted — let's meet every Monday morning at 7 AM to 'review your progress'.","acceptance",False,None),
("Employee","I'll bring my 'flowering enthusiasm' — and a very 'resilient' sense of humor.","submission",True,"sarcastic submission to the process"),
]},
{"scenario":"casual","topic":"sarcasm_car_repairs","utterances":[
("Owner","The mechanic told me that my 'minor oil leak' is actually a 'catastrophic failure of the entire engine block'.","rage",False,"unexpected and expensive car repair"),
("Friend","At least it's a 'failure of character' for the car — it's very 'dramatic' to break down in such a big way.","optimism",True,"sarcastic optimism about the car repair"),
("Owner","It's going to cost four thousand dollars and they're keeping the car for 'three weeks of observation'.","sadness",False,"high cost and loss of vehicle"),
("Friend","The car is 'in rehab' — it needs a 'period of reflection' to decide if it wants to be a vehicle anymore.","contempt",True,"sarcastic contempt for the situation"),
("Owner","I'm going to 'observe' the mechanic — by 'monitoring' his 'financial situation' from a very distance.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'investigative journalist' — we can write a 'gritty' expose on the 'secret life of spark plugs'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_food_trends","utterances":[
("Guest","They're charging fifteen dollars for 'deconstructed toast' which is just a piece of bread and a toaster.","disgust",False,"overpriced and pretentious food trend"),
("Friend","It's 'interactive dining' — you're 'participating in the creation of your own breakfast experience'.","contempt",True,"sarcastic defense of the food trend"),
("Guest","I'm 'participating' in being ripped off — I can 'interact' with my own toaster at home for free.","disapproval",False,"lack of appreciation for the 'trend'"),
("Friend","Your home toaster doesn't have the 'curated aesthetic' of this 'industrial-chic' environment.","pensiveness",True,"sarcastic pensiveness about the decor"),
("Guest","I'm going to start 'deconstructing' my payment — I'll give them a 'kit' of copper and paper.","aggressiveness",True,"sarcastic aggressive plan"),
("Friend","A 'monetary experience'! I'll be the 'fiscal consultant' — I'll tell them to 'assemble the value' themselves.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_corporate_retreats","utterances":[
("Employee","The 'mandatory team-building retreat' is a weekend of 'trust-falls' and 'collaborative yurt-building' in the woods.","annoyance",False,"unpopular and forced corporate retreat"),
("Colleague","I've always wanted to 'fall' into the arms of the guy who stole my lunch last Tuesday.","contempt",True,"sarcastic response to the retreat"),
("Employee","There's no Wi-Fi and we're supposed to 'connect with our inner leadership' through 'nature-based workshops'.","disgust",False,"lack of amenities and silly workshops"),
("Colleague","I'm sure the trees have a lot of 'actionable insights' on how to 'optimize our quarterly growth'.","pensiveness",True,"sarcastic pensiveness about the 'nature' aspect"),
("Employee","I'm going to 'connect' with a very 'nature-based' exit strategy — by walking until I find a road.","aggressiveness",True,"sarcastic aggressive response"),
("Colleague","I'll be your 'survival specialist' — I'll show you how to 'leverage the moss' for 'directional orientation'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_holiday_travel","utterances":[
("Traveller","The airport is currently a 'sea of forgotten luggage' and my flight has been delayed for the fourth time.","annoyance",False,"frustrating holiday travel experience"),
("Friend","It's a 'temporal buffer zone' — the airline is just giving you 'extra time to appreciate the architecture'.","optimism",True,"sarcastic optimism about the delay"),
("Traveller","I've been wearing the same shirt for two days and I'm pretty sure I'm starting to 'integrate' with the terminal.","grief",False,"discomfort due to long delay"),
("Friend","You're 'becoming the infrastructure' — it's a very 'holistic' way to travel, without actually moving.","contempt",True,"sarcastic contempt for the situation"),
("Traveller","I'm going to 'depart' this terminal — by 'launching' myself directly into the nearest hotel lobby.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'ground-based navigation' — I'll point at the 'exit' with 'great conviction'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_office_decor","utterances":[
("Admin","They've replaced the ergonomic chairs with 'wellness balls' to 'improve our core stability'.","annoyance",False,"uncomfortable office furniture change"),
("Developer","Perfect — because my 'core stability' is the only thing standing between me and a total mental breakdown.","contempt",True,"sarcastic response to the wellness balls"),
("Admin","I just saw the accountant bounce directly off his ball and into the water cooler.","amazement",True,"absurdity of the new furniture"),
("Developer","A 'dynamic hydration event'! It's a very 'collaborative' way to get a drink, if you think about it.","pensiveness",True,"sarcastic pensiveness about the accident"),
("Admin","I'm going to 'improve my stability' — by 'bolting' my old chair to the floor in the basement.","aggressiveness",True,"sarcastic aggressive response"),
("Developer","I'll be your 'structural engineer' — we can call it 'static wellness' and charge people for the silence.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_dog_training","utterances":[
("Owner","The 'positive reinforcement' trainer told me that I should 'thank the dog' every time he doesn't bite me.","disapproval",False,"weird and ineffective dog training advice"),
("Neighbor","It's 'gratitude-based obedience' — the dog is just 'waiting for his formal acknowledgement' before he complies.","contempt",True,"sarcastic mockery of the trainer"),
("Owner","He just 'thanked' the leg of the dining table by chewing it into a very 'minimalist' shape.","disgust",False,"dog's destructive behaviour"),
("Neighbor","The table was 'asking for a critique' — your dog is just a 'very honest furniture reviewer'.","pensiveness",True,"sarcastic pensiveness about the dog's 'critique'"),
("Owner","I'm going to 'thank' the dog — by 'escorting' him to a very 'positive' and 'reinforcing' kennel.","aggressiveness",True,"sarcastic aggressive response"),
("Neighbor","I'll be the 'logistics coordinator' — I'll hold the 'gratitude-based leash' for you.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_coffee_shops","utterances":[
("Customer","I just paid eight dollars for a 'charcoal latte' that looks and tastes like a very expensive puddle.","disgust",False,"bad and overpriced coffee trend"),
("Friend","It's 'activated' — you're not just drinking a latte, you're 'purifying your internal narrative'.","contempt",True,"sarcastic defense of the charcoal latte"),
("Customer","My 'internal narrative' is currently screaming about the loss of my eight dollars.","rage",False,"anger over wasted money"),
("Friend","The 'narrative' needs to 'embrace the bitterness' to achieve 'true enlightenment'.","pensiveness",True,"sarcastic pensiveness about the 'enlightenment'"),
("Customer","I'm going to 'activate' my own coffee at home — by 'igniting' the kitchen with my own frustration.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'fire-safety consultant' — I'll stand outside and 'curate the smoke'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_agile_transformation","utterances":[
("Developer","We're in the middle of a 'sprint' but our 'backlog' is currently longer than the actual roadmap.","annoyance",False,"poorly managed agile process"),
("Scrum_Master","It's a 'marathon of sprints' — we're just 'building the plane while we're falling out of the sky'.","optimism",True,"sarcastic optimism about the project's failure"),
("Developer","The 'daily stand-up' is currently forty-five minutes of sitting down and discussing our 'blockers' without fixing them.","disapproval",False,"inefficient agile rituals"),
("Scrum_Master","We're 'socializing the obstacles' — it's a very 'collaborative' way to stay exactly where we are.","contempt",True,"sarcastic response to the inefficiency"),
("Developer","I'm going to 'iterate' on my own career — by 'pivoting' to a company that doesn't use the word 'agile'.","aggressiveness",True,"sarcastic aggressive response"),
("Scrum_Master","I'll 'groom' your resignation letter — to make sure it's 'aligned' with our 'exit-strategy deliverables'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_baby_names","utterances":[
("Guest_A","They've decided to name the baby 'Quantum' because they want him to have a 'multi-dimensional future'.","disgust",False,"absurd and pretentious baby name"),
("Guest_B","A true 'observer-effect' — I'm sure he'll 'collapse' every conversation he's ever a part of.","contempt",True,"sarcastic mockery of the name"),
("Guest_A","The middle name is 'Synergy' — to ensure he's 'aligned with the corporate universe' from birth.","disapproval",False,"further absurdity in the name choice"),
("Guest_B","He's not a baby, he's a 'pre-revenue human asset' — it's a very 'forward-looking' parenting style.","pensiveness",True,"sarcastic pensiveness about the name"),
("Guest_A","I'm going to start naming my plants 'Entropy' and 'Regression' — to balance out the universe.","aggressiveness",True,"sarcastic aggressive plan"),
("Guest_B","I'll be the 'nomenclatural consultant' — I'll tell you if the 'vibe' is 'sufficiently pessimistic'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_gym_classes","utterances":[
("Member","The 'advanced yoga' class is just ninety minutes of 'visualizing your own breath' while a guy in the front row snores.","boredom",False,"unhelpful and boring gym class"),
("Friend","It's 'unconscious mindfulness' — the snoring is just a 'rhythmic anchor' for your internal focus.","optimism",True,"sarcastic optimism about the class"),
("Member","I just paid twenty dollars to 'experience the silence' in a room that smells like a very old gym bag.","disgust",False,"poor value and environment in the class"),
("Friend","The 'aroma' is 'organic' — it's a 'sensory reminder of the effort of previous generations'.","contempt",True,"sarcastic contempt for the gym"),
("Member","I'm going to 'visualize' my refund — it's a very 'challenging' pose for the manager.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'spiritual guide' — I'll show you how to 'transcend the contract' with 'great agility'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_innovation_labs","utterances":[
("Engineer","The company's 'innovation lab' just spent six months developing a 'smart' stapler that emails you when it's empty.","disapproval",False,"pointless 'innovation' in the workplace"),
("Colleague","Finally! My life's greatest mystery — the status of my stapler — has been 'digitally resolved'.","contempt",True,"sarcastic mockery of the smart stapler"),
("Engineer","It costs four hundred dollars and requires a 'monthly subscription' to the 'staple-tracking cloud'.","rage",False,"high cost and unnecessary subscription for basic tool"),
("Colleague","A 'staple-as-a-service' model — it's the 'disruption' we've all been waiting for in the stationary sector.","pensiveness",True,"sarcastic pensiveness about the 'disruption'"),
("Engineer","I'm going to 'innovate' a new way to use the stapler — by 'digitizing' it into the nearest trash can.","aggressiveness",True,"sarcastic aggressive response"),
("Colleague","I'll provide the 'technical support' — I'll stand there and 'document the impact' for our next quarterly review.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_house_parties","utterances":[
("Guest_A","The 'party' is currently six people sitting in a circle and staring at their own phones in silence.","boredom",False,"boring and socially awkward party"),
("Guest_B","It's a 'digital-first social experience' — we're 'interacting' through the 'shared medium of the internet'.","contempt",True,"sarcastic response to the lack of social interaction"),
("Guest_A","The 'host' just asked if we could 'all follow his new Instagram account' for 'exclusive party updates'.","disgust",False,"self-promotional behavior from the host"),
("Guest_B","The 'exclusive updates' are probably just photos of us staring at our phones — it's very 'meta'.","pensiveness",True,"sarcastic pensiveness about the situation"),
("Guest_A","I'm going to 'follow' the nearest exit — it's a very 'exclusive' and 'private' path for me right now.","aggressiveness",True,"sarcastic aggressive response"),
("Guest_B","I'll be your 'social-media strategist' — I'll tell the host that your 'engagement' was 'off the charts'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_tech_conferences","utterances":[
("Attendee","I just paid fifteen hundred dollars to hear a guy in a hoodie talk about how 'failure is the new success'.","annoyance",False,"expensive and cliché tech conference"),
("Friend","A 'pioneering perspective'! I'm sure his 'success at failing' is very inspirational to your bank account.","contempt",True,"sarcastic mockery of the speaker"),
("Attendee","The 'networking lunch' was a single cold taco and a bottle of 'artisanally sourced' air.","disgust",False,"poor value for money at the conference"),
("Friend","The taco was 'minimalist' — to reflect the 'lean-startup' philosophy of the catering team.","pensiveness",True,"sarcastic pensiveness about the taco"),
("Attendee","I'm going to 'fail' to pay my registration fee for next year — to see if it makes me more 'successful'.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'brand ambassador' — I'll tell everyone that your 'absence' is a 'bold strategic pivot'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_it_upgrades","utterances":[
("User","IT just 'upgraded' our software and now the 'save' button has been replaced by a 'mystery icon' that looks like a cat.","annoyance",False,"unnecessary and confusing software change"),
("Colleague","It's a 'user-centric interface' — the cat represents the 'playful nature of data integrity'.","contempt",True,"sarcastic response to the UI change"),
("User","I just 'deleted' an entire project because the 'cat icon' was actually a 'purge-all-data' function.","rage",False,"negative impact of the software 'upgrade'"),
("Colleague","The cat was 'hungry' for your data — it's a very 'interactive' way to learn about the value of backups.","optimism",True,"sarcastic optimism about the loss"),
("User","I'm going to 'upgrade' the IT office — by 'installing' a very 'user-centric' lock on their door.","aggressiveness",True,"sarcastic aggressive response"),
("Colleague","I'll be your 'systems administrator' — I'll tell them the 'key' is a 'metaphor for their own competence'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_farmers_markets","utterances":[
("Shopper","They're charging twelve dollars for a 'rustic' carrot that still has the 'original soil' attached for 'authenticity'.","disgust",False,"overpriced and pretentious food at a market"),
("Friend","The soil is 'locally sourced' — you're not just buying a carrot, you're 'acquiring a piece of the terroir'.","contempt",True,"sarcastic defense of the expensive carrot"),
("Shopper","I can 'acquire a piece of the terroir' from my own backyard for free — and without the 'rustic' price tag.","disapproval",False,"lack of appreciation for the 'authenticity'"),
("Friend","Your backyard doesn't have the 'curated experience' of a bearded man in a flannel shirt judging your choices.","pensiveness",True,"sarcastic pensiveness about the market"),
("Shopper","I'm going to start 'curating' my own dirt — and selling it to people who want a 'visceral connection' to my lawn.","aggressiveness",True,"sarcastic aggressive plan"),
("Friend","A 'landscape-as-a-service' model! I'll be the 'soil specialist' — I'll tell them it's 'pre-aged'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_office_efficiency","utterances":[
("Manager","We're introducing a 'no-talking' policy between 9 AM and 11 AM to 'optimize our cognitive throughput'.","annoyance",False,"unpopular and restrictive office policy"),
("Employee","Brilliant — because my 'cognitive throughput' really thrives in a room full of people breathing loudly in silence.","contempt",True,"sarcastic response to the no-talking policy"),
("Manager","It's a 'monastic approach to productivity' and we expect to see a 'transcendental increase' in output.","trust",False,"manager using weirdly spiritual language for work"),
("Employee","I've already 'transcended' the need for a salary — I'm currently 'outputting' my own existential dread.","disapproval",True,"sarcastic disapproval of the policy"),
("Manager","I appreciate your 'monastic dedication' — let's review the 'spiritual deliverables' on Friday.","acceptance",False,None),
("Employee","I'll bring my 'vow of silence' — to my next performance review.","submission",True,"sarcastic submission to the policy"),
]},
{"scenario":"casual","topic":"sarcasm_car_insurance","utterances":[
("Owner","My car insurance just went up by thirty percent because of 'increased regional probability of gravitational anomalies'.","annoyance",False,"ridiculous excuse for insurance price hike"),
("Friend","Very thorough! They're just 'protecting you from the unexpected collapse of the laws of physics'.","optimism",True,"sarcastic optimism about the price hike"),
("Owner","I'm pretty sure 'gravitational anomalies' is just code for 'we want more of your money'.","rage",False,"anger over the unjustified increase"),
("Friend","The 'anomalies' are very expensive to monitor — the insurance company needs a 'quantum-ready' balance sheet.","contempt",True,"sarcastic contempt for the insurance company"),
("Owner","I'm going to 'anomalize' my next payment — by 'drifting' it into a completely different bank account.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'physics consultant' — I'll tell them the 'payment' is currently 'trapped in a singularity'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_art_galleries","utterances":[
("Guest","The 'art' is just a series of blurry photos of the artist's own feet 'exploring the concept of movement'.","boredom",False,"judging pretentious and low-quality art"),
("Friend","It's 'The Pedestrian Odyssey' — the blurry feet are a 'metaphor for the instability of the human journey'.","contempt",True,"sarcastic defense of the art"),
("Guest","I've seen more 'instability' in a bowl of jelly — and it had a better 'narrative arc' too.","disapproval",False,"lack of appreciation for the 'art'"),
("Friend","The jelly was probably 'un-curated' — it lacked the 'intellectual rigor' of these blurry toes.","pensiveness",True,"sarcastic pensiveness about the art"),
("Guest","I'm going to 'rigorously' walk out of here — and 'explore the concept' of a very bright exit sign.","aggressiveness",True,"sarcastic aggressive plan"),
("Friend","A 'departure-based performance'! I'll be the 'curator of your exit' — I'll tell everyone it was 'unscripted'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_office_temperature","utterances":[
("Employee","The office is currently so cold that my coffee has actually turned into a very small iceberg.","annoyance",False,"extremely cold office temperature"),
("Colleague","It's an 'environmental-efficiency strategy' — they want to 'freeze our productivity' into a more stable state.","contempt",True,"sarcastic response to the cold"),
("Employee","I'm currently wearing three sweaters and a scarf and I can still see my own breath 'collaborating' with the air.","rage",False,"physical discomfort due to cold"),
("Colleague","Your breath is just 'manifesting its own workspace' — it's a very 'dynamic' way to experience the HVAC system.","optimism",True,"sarcastic optimism about the situation"),
("Employee","I'm going to 'collaborate' with the thermostat — by 'introducing' it to a very warm and 'innovative' lighter.","aggressiveness",True,"sarcastic aggressive response"),
("Colleague","I'll be your 'thermal specialist' — I'll stand in front of the smoke detector and 'curate the alarm'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_fast_food","utterances":[
("Customer","I just ordered a 'gourmet burger' and it looks like it was 'deconstructed' by a very angry lawnmower.","disgust",False,"badly made fast food burger"),
("Friend","It's 'rustic' — the 'irregular geometry' of the bun is a 'statement on the unpredictability of the grain'.","contempt",True,"sarcastic defense of the bad burger"),
("Customer","The 'unpredictability of the grain' tastes suspiciously like a 'lack of basic kitchen skills'.","disapproval",False,"lack of quality in the food"),
("Friend","The 'skills' are just 'too advanced' for your 'linear expectations' of what a burger should look like.","pensiveness",True,"sarcastic pensiveness about the burger"),
("Customer","I'm going to 'deconstruct' my relationship with this restaurant — by 'reconstructing' my dinner elsewhere.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'culinary strategist' — I'll point at the 'golden arches' with 'great conviction'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_pet_etiquette","utterances":[
("Guest","She brought her 'emotional support' parrot to a funeral and it spent the whole time mimicking the priest.","disgust",False,"inappropriate pet behavior in a somber setting"),
("Friend","It's 'vocal solidarity' — the parrot is just 'amplifying the spiritual message' for the benefit of the back row.","contempt",True,"sarcastic mockery of the bird and owner"),
("Guest","The 'spiritual message' was mostly about 'wanting a cracker' and 'being a pretty boy'.","amazement",True,"absurdity of the bird's interruptions"),
("Friend","A true 'multi-species dialogue' — it's a very 'forward-looking' approach to grieving.","pensiveness",True,"sarcastic pensiveness about the situation"),
("Guest","I'm going to 'support' the parrot — by 'escorting' it to a very 'quiet' and 'reflective' birdcage.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be the 'liturgical coordinator' — I'll tell everyone that the 'cracker' was a 'metaphor for the soul'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_office_parties","utterances":[
("Employee","The 'company holiday party' is currently a bowl of generic pretzels and a single lukewarm bottle of grape juice.","boredom",False,"unexciting office party"),
("Colleague","It's a 'fiscally responsible celebration' — they want us to 'focus on the value of our connections' instead of the food.","contempt",True,"sarcastic response to the poor party"),
("Employee","I'm 'connecting' with the fact that my bonus was apparently 'integrated' into this bowl of pretzels.","rage",False,"anger over poor rewards and party quality"),
("Colleague","The pretzels are 'artisanal' — if you use enough imagination and a very 'creative' mindset.","optimism",True,"sarcastic optimism about the snacks"),
("Employee","I'm going to 'celebrate' my own way — by 'optimizing' my 'exit-time deliverable' right now.","aggressiveness",True,"sarcastic aggressive response"),
("Colleague","I'll be your 'party strategist' — I'll tell everyone that your 'departure' was a 'bold artistic statement'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_smart_watches","utterances":[
("User","My smart watch just told me that I've been 'sitting for too long' while I'm currently running on a treadmill.","annoyance",False,"unreliable smart watch notifications"),
("Friend","The watch is just 'challenging your perception of movement' — it's a very 'existential' piece of technology.","contempt",True,"sarcastic response to the watch's error"),
("User","I've achieve 'peak frustration' — my heart rate is 140 and the watch thinks I'm 'napping'.","rage",False,"further inaccuracy of the smart watch"),
("Friend","A 'high-intensity nap'! You're 'revolutionizing the science of recovery' without even knowing it.","optimism",True,"sarcastic optimism about the situation"),
("User","I'm going to 'notify' the watch — by 'syncing' it with the bottom of a very deep lake.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'technical advisor' — I'll help you 'calibrate the impact' with a very heavy stone.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_art_critics","utterances":[
("Guest","The 'art critic' spent twenty minutes talking about the 'gestural integrity' of a spilled coffee stain on the floor.","boredom",False,"witnessing pretentious art criticism"),
("Friend","A 'masterpiece of accidental domesticity'! I'm sure the stain is 'deeply honored' by his 'analytical gaze'.","contempt",True,"sarcastic mockery of the critic"),
("Guest","I'm pretty sure it's just a 'masterpiece' of 'not knowing how to hold a cup' while walking.","disapproval",False,"literalist view of the 'art'"),
("Friend","Your 'lack of vision' is so 'pedestrian' — you're missing the 'subtle dialogue' between the caffeine and the tile.","pensiveness",True,"sarcastic pensiveness about the 'art'"),
("Guest","I'm going to 'dialogue' with the critic — by 'critiquing' his 'gestural integrity' while I push him toward the exit.","aggressiveness",True,"sarcastic aggressive plan"),
("Friend","A 'confrontational performance art piece'! I'll be the 'curator of the conflict' — it'll be 'seminal'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_it_maintenance","utterances":[
("User","IT just scheduled a 'critical system maintenance' for the exact window of my most important client call.","annoyance",False,"poorly timed IT maintenance"),
("Colleague","They want to 'test your resilience' — a client call without a working computer is a 'true leadership challenge'.","contempt",True,"sarcastic response to the bad timing"),
("User","I'm currently 'practicing leadership' by screaming into a very 'critical' and 'offline' headset.","rage",False,"frustration due to IT issues"),
("Colleague","The screaming is 'unfiltered communication' — it's a very 'honest' way to 'socialize the problem'.","optimism",True,"sarcastic optimism about the situation"),
("User","I'm going to 'maintain' the IT department — by 'scheduling' a very 'critical' meeting with their front door lock.","aggressiveness",True,"sarcastic aggressive response"),
("Colleague","I'll be your 'logistics specialist' — I'll tell them the 'offline status' is a 'metaphor for their career prospects'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_parking_lots","utterances":[
("Driver","I've been driving in circles for twenty minutes and the only 'available' space is currently occupied by a shopping cart.","annoyance",False,"frustrating parking experience"),
("Friend","The cart is just 'holding the space' for a 'very important groceries' — it's a 'high-priority vehicle'.","contempt",True,"sarcastic response to the shopping cart"),
("Driver","I'm about to 'integrate' the shopping cart into the nearest 'unauthorized-parking-zone' bush.","rage",False,"anger at the blocked parking space"),
("Friend","A 'structural relocation project'! I'm sure the cart will appreciate the 'new environmental context'.","optimism",True,"sarcastic optimism about the move"),
("Driver","I'm going to 'park' my own frustration — by 'launching' it directly into the manager's office.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'parking consultant' — I'll point at the 'fire lane' with 'great existential conviction'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_mentorship_v2","utterances":[
("Junior","My mentor told me that 'crying in the bathroom' is a 'valid professional development milestone'.","annoyance",False,"toxic advice about handling workplace stress"),
("Senior","He's right — the 'acoustics of the second stall' are specifically designed for 'emotional throughput'.","contempt",True,"sarcastic response to the bad advice"),
("Junior","He also said that 'boundaries are just obstacles to your full potential as a human resource'.","fear",False,"further toxic advice about overwork"),
("Senior","I'm sure your 'full potential' will look great on the 'exhaustion-based performance review'.","pensiveness",True,"sarcastic pensiveness about the consequences"),
("Junior","I'm going to 'mentor' my mentor — by 'developing' a 'plan' to 'optimize his distance' from my desk.","aggressiveness",True,"sarcastic aggressive response"),
("Senior","I'll be your 'career coach' — I'll help you 'leverage the silence' of his absence.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_baby_showers","utterances":[
("Guest_A","The 'baby shower' activity is currently 'guessing the flavor of baby food' while blindfolded.","boredom",False,"judging boring and weird baby shower activities"),
("Guest_B","A 'culinary investigation'! I'm sure the 'mystery mush' is 'deeply revealing' of our shared humanity.","contempt",True,"sarcastic mockery of the activity"),
("Guest_A","I just 'guessed' that the flavor is 'regret' and the host told me I'm 'not being supportive'.","disgust",False,"negative reaction to the activity and the host's response"),
("Guest_B","'Regret' is a very 'nuanced flavor profile' — it's a 'statement on the complexity of adult life'.","pensiveness",True,"sarcastic pensiveness about the 'flavor'"),
("Guest_A","I'm going to 'support' the next activity — by 'blindfolding' myself and 'guessing' where the exit is.","aggressiveness",True,"sarcastic aggressive plan"),
("Guest_B","I'll be your 'exit strategist' — I'll tell everyone that your 'departure' was a 'bold sensory experiment'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_delivery_apps_v2","utterances":[
("User","My delivery app just told me that my 'courier is currently taking a detour to appreciate the scenery'.","annoyance",False,"ridiculous excuse for delivery delay"),
("Friend","He's a 'mobile landscape enthusiast' — your burger is just a 'passive participant' in his artistic vision.","contempt",True,"sarcastic response to the delay"),
("User","The 'scenery' is currently a 'construction site' and my burger is 'experiencing the dust' for forty minutes.","rage",False,"further frustration due to delay and location"),
("Friend","The dust adds 'texture' — it's a 'multidimensional flavor experience' that you're not paying enough for.","optimism",True,"sarcastic optimism about the bad food"),
("User","I'm going to 'detour' my payment — into a very 'private' and 'inaccessible' part of my bank account.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'fiscal navigator' — I'll tell the app that the 'funds' are 'currently appreciating the local economy'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_it_policies_v2","utterances":[
("User","IT just banned 'all external software' including the 'calculator' because it was a 'security risk'.","annoyance",False,"excessive and restrictive IT policy"),
("Colleague","They want to 'return to our intellectual roots' — using your brain is the 'ultimate secure infrastructure'.","contempt",True,"sarcastic response to the calculator ban"),
("User","I'm currently 'calculating' my budget for next month using a 'high-performance' piece of chalk and a rock.","submission",True,"sarcastic submission to the restriction"),
("Colleague","A 'lithographic approach to accounting'! It's very 'hardened' against 'digital intrusion'.","pensiveness",True,"sarcastic pensiveness about the 'new system'"),
("User","I'm going to 'secure' the IT manager's laptop — by 'manually' and 'permanently' disconnecting it from reality.","aggressiveness",True,"sarcastic aggressive response"),
("Colleague","I'll be your 'hardware specialist' — I'll stand there and 'verify the physical separation'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_modern_parenting","utterances":[
("Neighbor","She's raised her toddler to be 'choice-driven' and now he refuses to put on pants because it's 'not his truth'.","disgust",False,"judging permissive and weird parenting styles"),
("Friend","His 'truth' is currently 'maximal ventilation' — it's a very 'honest' and 'unfiltered' way to live.","contempt",True,"sarcastic mockery of the parenting"),
("Neighbor","The 'choice-driven' child just chose to 'redesign' the hood of my car with a very sharp rock.","rage",False,"child's destructive behavior resulting from parenting style"),
("Friend","He's a 'mobile street artist' — your car was just a 'blank canvas' for his 'unfiltered expression'.","amazement",True,"sarcastic amazement at the child's 'choice'"),
("Neighbor","I'm going to 'choice-drive' him — to a very 'choice-limited' environment, like a very small playpen.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'pedagogical consultant' — I'll tell his mother that the 'playpen' is a 'meditation chamber'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_public_libraries","utterances":[
("Visitor","The library just replaced the 'quiet zone' with a 'collaborative-media-hub' that sounds like an arcade.","annoyance",False,"unpopular change to library environment"),
("Friend","They want to 'democratize the silence' — by 'integrating' it into a 'sea of high-decibel innovation'.","contempt",True,"sarcastic response to the noise"),
("Visitor","I'm currently 'collaborating' with a 'media-hub' that's just a guy playing 'drums' on his own desk.","disapproval",False,"lack of quiet in the library"),
("Friend","He's a 'rhythmic scholar' — he's 'translating the Dewey Decimal system' into a 'percussive narrative'.","pensiveness",True,"sarcastic pensiveness about the drummer"),
("Visitor","I'm going to 'democratize' his drumsticks — by 'integrating' them into the nearest 'non-collaborative' trash can.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'librarian of the void' — I'll point at the 'shushing sign' with 'great existential weight'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_it_helpdesk_v2","utterances":[
("User","IT told me that my 'ticket is currently in a queue' but the 'queue' is actually just a 'black hole' on their website.","annoyance",False,"unhelpful and deceptive IT support"),
("Colleague","The 'black hole' is a 'feature' — it's where 'unsolvable problems' go to 'achieve a state of eternal stability'.","contempt",True,"sarcastic response to the IT ticket system"),
("User","I just 'stable-ized' my computer by 'restarting' it fifteen times until the 'black hole' decided to let go.","submission",True,"sarcastic submission to the technical failure"),
("Colleague","A 'manual-override strategy'! It's very 'high-touch' and 'independent' of the 'corporate support infrastructure'.","optimism",True,"sarcastic optimism about the 'fix'"),
("User","I'm going to 'queue' my own 'maintenance' — by 'prioritizing' a very 'offline' lunch for the rest of the day.","aggressiveness",True,"sarcastic aggressive response"),
("Colleague","I'll be your 'lunch coordinator' — I'll tell the 'black hole' that you're 'out for a spiritual sync'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_art_auctions","utterances":[
("Guest","They just sold a 'canvas of pure air' for six million dollars to a guy who already owns 'the concept of blue'.","disgust",False,"absurd and overpriced art market"),
("Friend","A true 'visionary'! I'm sure the 'air' has a very 'substantive' presence in his 'metaphysical portfolio'.","contempt",True,"sarcastic mockery of the buyer"),
("Guest","I'm pretty sure the 'metaphysical portfolio' is just code for 'more money than actual common sense'.","disapproval",False,"judging the wealth and lack of sense in the art world"),
("Friend","Common sense is 'so last century' — in the art world, 'absurdity' is the only 'convertible currency'.","pensiveness",True,"sarcastic pensiveness about the art market"),
("Guest","I'm going to start 'auctioning' my own 'intentions' — and see if I can get 'the concept of wealth' in return.","aggressiveness",True,"sarcastic aggressive plan"),
("Friend","I'll be your 'auctioneer of the abstract' — I'll tell everyone that your 'greed' is actually 'curated ambition'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_weather_apps_v2","utterances":[
("User","My weather app said it would be 'sunny and warm' and I'm currently 'standing in a monsoon' in a t-shirt.","annoyance",False,"inaccurate weather app forecast"),
("Friend","The app is just 'projecting the weather it wants for you' — it's a very 'supportive' and 'optimistic' AI.","optimism",True,"sarcastic optimism about the app's error"),
("User","I'm currently 'experiencing the optimism' by 'absorbing three inches of rain' into my own shoes.","rage",False,"physical discomfort due to bad forecast"),
("Friend","The 'absorption' is 'organic hydration' — you're 'becoming one with the local water table'.","contempt",True,"sarcastic contempt for the situation"),
("User","I'm going to 'project' a new 'forecast' — by 'deleting' the app and 'replacing' it with a very reliable window.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'meteorological consultant' — I'll tell you if it's 'raining' with 'great manual precision'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_office_ethics","utterances":[
("Employee","The 'ethics training' was a forty-minute video on how to 'report your coworkers for excessive happiness'.","annoyance",False,"weird and suspicious workplace training"),
("Colleague","They want to 'ensure a consistent level of professional misery' — it's very 'on-brand' for the current leadership.","contempt",True,"sarcastic response to the training"),
("Employee","I just 'reported' myself for 'feeling a brief moment of hope' during the morning coffee run.","submission",True,"sarcastic submission to the toxic culture"),
("Colleague","A 'self-correction strategy'! It's very 'proactive' and shows you're 'aligned with the corporate gloom'.","optimism",True,"sarcastic optimism about the 'report'"),
("Employee","I'm going to 'ethics-train' the HR department — by 'documenting' their 'excessive use of meaningless jargon'.","aggressiveness",True,"sarcastic aggressive response"),
("Colleague","I'll be your 'compliance officer' — I'll tell everyone that your 'rage' is actually 'heightened ethical awareness'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_baby_shower_v2","utterances":[
("Guest_A","The 'baby shower' is now a 'pre-natal networking event' where we're supposed to 'swap corporate synergies' for the toddler.","disgust",False,"absurd and overly corporate social event"),
("Guest_B","A 'strategic start'! I'm sure the 'toddler' will appreciate the 'aligned leadership' of his toy box.","contempt",True,"sarcastic mockery of the event theme"),
("Guest_A","I just 'swapped a synergy' for a 'diaper-changing workflow' and the host told me I'm 'not being a team player'.","disapproval",False,"negative reaction to the corporate theme"),
("Guest_B","The 'team' is currently 'under-resourced' in the 'naptime department' — it's a very 'dynamic' nursery.","pensiveness",True,"sarcastic pensiveness about the situation"),
("Guest_A","I'm going to 'network' my way to the nearest exit — it's a very 'high-priority' and 'un-synergized' path for me.","aggressiveness",True,"sarcastic aggressive response"),
("Guest_B","I'll be your 'exit-strategy consultant' — I'll tell everyone that your 'departure' was a 'bold career pivot'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"casual","topic":"sarcasm_smart_kitchens","utterances":[
("Owner","My 'smart' fridge just sent me a 'critical alert' that my 'milk is currently experiencing an identity crisis'.","annoyance",False,"absurd and unhelpful smart appliance notification"),
("Friend","It's 'self-aware dairy' — your milk is just 'questioning its role in the breakfast hierarchy'.","optimism",True,"sarcastic response to the fridge's alert"),
("Owner","I'm currently 'dialoguing' with a 'carton of soy milk' about its 'life goals' while I'm trying to make cereal.","submission",True,"sarcastic submission to the absurdity"),
("Friend","The 'cereal' is 'post-revenue' — it's just 'waiting for the milk to find its existential stability'.","contempt",True,"sarcastic contempt for the 'smart' tech"),
("Owner","I'm going to 'identify' the fridge's future — by 'unplugging' it and 'restarting' it in a landfill.","aggressiveness",True,"sarcastic aggressive response"),
("Friend","I'll be your 'appliance specialist' — I'll tell the 'milk' that its 'crisis' was 'successfully localized'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"workplace","topic":"sarcasm_it_maintenance_v3","utterances":[
("User","IT just 'upgraded' my laptop and now the 'power' button is a 'virtual experience' that requires a four-digit PIN.","annoyance",False,"unnecessary and frustrating hardware change"),
("Colleague","They want to 'ensure intentional usage' — turning on your computer is a 'privileged interaction'.","contempt",True,"sarcastic response to the PIN requirement"),
("User","I just 'authenticated' my own 'intention' by 'typing the PIN' forty times until the laptop decided to 'wake up'.","submission",True,"sarcastic submission to the technical barrier"),
("Colleague","A 'rigorous access-control strategy'! It's very 'secure' against 'accidental productivity'.","optimism",True,"sarcastic optimism about the 'fix'"),
("User","I'm going to 'intentionally' use the laptop — by 'placing' it in the 'deep-freeze' of the IT storage room.","aggressiveness",True,"sarcastic aggressive response"),
("Colleague","I'll be your 'hardware auditor' — I'll tell everyone that the 'cooling' was a 'strategic performance boost'.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"sarcasm_modern_weddings_v2","utterances":[
("Guest_A","The 'wedding' is currently a 'silent-disco' where everyone is wearing headphones and staring at a neon sign.","boredom",False,"judging weird and isolating wedding trends"),
("Guest_B","It's an 'un-mediated auditory experience' — we're 'connected' through the 'invisible medium of sound'.","contempt",True,"sarcastic response to the silent disco"),
("Guest_A","The 'neon sign' just told me to 'vibrate with the frequency of the union' — I'm pretty sure I'm just vibrating with hunger.","disgust",False,"absurd and pretentious wedding messaging"),
("Guest_B","The 'hunger' is just 'vibrational potential' — you're 'preparing your soul' for the 'monogrammed popcorn'.","pensiveness",True,"sarcastic pensiveness about the situation"),
("Guest_A","I'm going to 'vibrate' my way to the nearest hotel bar — it's a very 'high-frequency' and 'populated' space for me.","aggressiveness",True,"sarcastic aggressive response"),
("Guest_B","I'll be your 'auditory navigator' — I'll tell the bride that your 'frequency' was 'too advanced' for the headphones.","joy",True,"sarcastic joy at the plan"),
]},
{"scenario":"social","topic":"admiration_skill","utterances":[
("Mentor","I watched your live coding session and your ability to refactor that legacy mess in real-time was stunning.","admiration",False,"genuine admiration for technical skill"),
("Junior","I was mostly just trying not to panic while three hundred people watched me forget how to use a for-loop.","apprehension",False,"imposter syndrome and performance anxiety"),
("Mentor","The way you handled the edge cases showed a depth of understanding that most senior developers lack.","admiration",False,"further affirmation of skill"),
("Junior","I've spent the last six months practically living in the documentation for that specific module.","vigilance",False,"diligent preparation acknowledged"),
("Mentor","It shows — you've achieved a level of mastery that is genuinely inspiring to the rest of the team.","admiration",False,"skill recognized as inspiration for others"),
("Junior","I really appreciate that feedback — it makes all those late nights feel significantly more worth it.","joy",False,"relief and joy at recognition"),
]},
{"scenario":"casual","topic":"terror_near_miss","utterances":[
("Driver","A truck drifted directly into my lane on the highway and I missed it by literally three inches.","terror",False,"narrow escape from high-speed collision"),
("Friend","Did you manage to pull over safely or are you still driving while processing the shock.","apprehension",False,"concern for friend's immediate safety"),
("Driver","I'm sitting in a gas station parking lot and my hands won't stop shaking enough to hold a water bottle.","terror",False,"visceral physical response to extreme danger"),
("Friend","Stay exactly where you are — I am leaving my office right now to come and get you.","trust",False,"immediate practical support offered"),
("Driver","I saw the grill of the truck and I genuinely thought that was the last thing I would ever see.","terror",False,"encounter with mortality"),
("Friend","It wasn't. You are safe, you are alive, and I will be there in ten minutes.","serenity",False,"calming presence and reassurance"),
]},
{"scenario":"conflict","topic":"loathing_betrayal","utterances":[
("Partner_A","I found the emails you've been sending to your 'ex' while telling me you were working late.","loathing",False,"discovery of infidelity and deception"),
("Partner_B","It's not what it looks like — we were just catching up on some old mutual business.","contempt",True,"weak and deceptive excuse"),
("Partner_A","The level of casual dishonesty you've displayed over the last month is genuinely stomach-turning.","loathing",False,"revulsion at partner's character"),
("Partner_B","You're being dramatic — it was just a few messages, it didn't mean anything at all.","disapproval",False,"dismissive reaction to valid anger"),
("Partner_A","I look at you now and I don't even recognize the person I thought I was building a life with.","loathing",False,"fundamental loss of respect and connection"),
("Partner_B","Fine, if you're going to be like this, maybe I should just leave right now.","aggressiveness",False,"defensive threat to end the interaction"),
]},
{"scenario":"casual","topic":"serenity_morning","utterances":[
("Person_A","I woke up at five and watched the fog lift off the lake with a cup of coffee and total silence.","serenity",False,"peaceful morning experience"),
("Person_B","That sounds like a scene from a movie — I haven't seen five AM without a deadline in years.","interest",False,"curiosity about peaceful routine"),
("Person_A","There was no phone, no email, just the sound of the water and the gradual return of the light.","serenity",False,"digital detox and nature connection"),
("Person_B","I can almost feel my blood pressure dropping just hearing you describe it.","acceptance",False,"empathetic calm"),
("Person_A","I think I've forgotten that the world can be this quiet and this completely undemanding.","serenity",False,"re-discovery of stillness"),
("Person_B","We should all probably spend more time watching fog and less time watching notifications.","optimism",False,"positive reflection on lifestyle change"),
]},
{"scenario":"social","topic":"apprehension_exam","utterances":[
("Student_1","I'm standing outside the exam hall and I've suddenly forgotten every single thing I studied.","apprehension",False,"pre-exam panic and memory loss"),
("Student_2","That's just the adrenaline — the information is still in there, you just can't see it through the fog.","trust",False,"reassurance from a peer"),
("Student_1","What if they ask about the third-century economic reforms — I only skimmed that chapter once.","apprehension",False,"specific fear of knowledge gaps"),
("Student_2","If they ask about that, everyone else is going to be in the exact same boat as you.","acceptance",False,"normalizing the difficulty"),
("Student_1","I can feel my heart pounding in my ears — I'm pretty sure I'm about to pass out before the doors open.","apprehension",False,"physical symptoms of intense anxiety"),
("Student_2","Breathe with me — four counts in, four counts out. We are going to get through this.","serenity",False,"active calming exercise"),
]},
{"scenario":"workplace","topic":"distraction_noise","utterances":[
("Worker_A","I've been trying to write this report for two hours but the construction outside is making it impossible.","distraction",False,"environmental noise preventing focus"),
("Worker_B","Is it the drilling or the sound of the workers shouting at each other that's worse.","interest",False,"seeking specific detail about the distraction"),
("Worker_A","It's the unpredictable rhythm of it — the moment I start a sentence, there's a massive crash.","distraction",False,"unpredictable stimuli breaking concentration"),
("Worker_B","I have some noise-cancelling headphones you can borrow if you think they'll help at all.","trust",False,"practical help offered"),
("Worker_A","I've tried headphones but I can still feel the vibration of the heavy machinery in my desk.","distraction",False,"multi-sensory distraction"),
("Worker_B","Maybe we should just move the 'working group' to the coffee shop across the street for the afternoon.","optimism",False,"proposing a constructive alternative"),
]},
{"scenario":"casual","topic":"serenity_garden","utterances":[
("Gardener","I spent the afternoon weeding the flower beds and I didn't think about work even once.","serenity",False,"therapeutic manual labor"),
("Friend","There's something about getting your hands in the dirt that really grounds the mind, isn't there.","trust",False,"shared understanding of gardening value"),
("Gardener","The bees were working on the lavender and the sun was just warm enough without being hot.","serenity",False,"sensory peace in nature"),
("Friend","I'm genuinely jealous — my afternoon was spent in a series of increasingly loud Zoom calls.","sadness",False,"contrast with stressful workday"),
("Gardener","I'll bring you a bouquet of the sweet peas tomorrow — they smell like pure happiness.","love",False,"kind gesture toward a friend"),
("Friend","That would be amazing — I'll put them right next to my laptop to remind me that the outside exists.","joy",False,"anticipatory joy"),
]},
{"scenario":"social","topic":"admiration_volunteer","utterances":[
("Organizer","She has been at the shelter every single Saturday for five years without missing a day.","admiration",False,"long-term commitment recognized"),
("Visitor","Five years is a staggering amount of time to give to a cause so consistently.","amazement",False,"magnitude of effort recognized"),
("Organizer","She doesn't just do the work — she knows the names and stories of every person who comes through the door.","admiration",False,"quality of care praised"),
("Visitor","That kind of empathy is rare — most people would have burned out after the first six months.","trust",False,"acknowledging the difficulty of the role"),
("Organizer","She says it's not a sacrifice — it's the part of her week where she feels most like herself.","admiration",False,"selfless motivation admired"),
("Visitor","I'd like to sign up for the orientation session — I think I need to learn from her.","interest",False,"desire to emulate the admired behavior"),
]},
{"scenario":"casual","topic":"terror_turbulence","utterances":[
("Passenger_A","The plane just dropped three hundred feet in two seconds and the oxygen masks actually deployed.","terror",False,"severe aircraft turbulence incident"),
("Passenger_B","I'm holding onto the armrests so hard I think I've actually cracked the plastic.","terror",False,"physical manifestation of extreme fear"),
("Passenger_A","The screaming from the back of the plane is making the whole thing feel ten times worse.","terror",False,"contagious panic in a crisis"),
("Passenger_B","I've never been this scared in my entire life — I just want to be on the ground.","terror",False,"desperate desire for safety"),
("Passenger_A","Look at me — just keep looking at me and keep breathing. The pilot says we are clear of it now.","trust",False,"peer support during extreme stress"),
("Passenger_B","My chest feels like it's in a vice — I can't believe we are still in the air.","terror",False,"lingering physiological shock"),
]},
{"scenario":"conflict","topic":"loathing_scam","utterances":[
("Victim","He convinced my eighty-year-old mother to give him her entire life savings for a 'protection plan'.","loathing",False,"revulsion at exploitation of the elderly"),
("Officer","We've been tracking this specific individual for months — he targets the most vulnerable people.","vigilance",False,"law enforcement focus on the perpetrator"),
("Victim","To look someone that age in the eye and systematically rob them is a level of evil I can't grasp.","loathing",False,"moral revulsion at the perpetrator's actions"),
("Officer","We have enough evidence now to ensure he won't be doing this to anyone else for a very long time.","trust",False,"reassurance of justice"),
("Victim","I hope he spends every day in prison thinking about the people whose lives he's destroyed.","loathing",False,"sustained loathing and desire for accountability"),
("Officer","We'll need your mother's formal statement tomorrow to finalize the charges.","acceptance",False,"procedural next step"),
]},
{"scenario":"social","topic":"apprehension_result","utterances":[
("Candidate","The email with the final decision is in my inbox and I've been staring at it for twenty minutes.","apprehension",False,"fear of life-altering news"),
("Friend","Do you want me to open it for you and just tell you the result so you don't have to look.","trust",False,"offer of support in a stressful moment"),
("Candidate","No, I need to be the one — but my stomach is currently doing somersaults and I'm dizzy.","apprehension",False,"physical symptoms of high-stakes anxiety"),
("Friend","Whatever it says, we have a plan for the next step. You aren't alone in this.","trust",False,"reassurance of companionship regardless of outcome"),
("Candidate","I've put so much of my identity into this role — if it's a no, I don't know who I am tomorrow.","apprehension",False,"identity-linked anxiety"),
("Friend","You are still you. The role is a job, not your soul. Open it when you're ready.","serenity",False,"perspective-shifting calm"),
]},
{"scenario":"workplace","topic":"distraction_open_plan","utterances":[
("Employee_A","I can hear three different sales calls and a birthday celebration while I'm trying to code.","distraction",False,"multiple auditory distractions in office"),
("Manager","The open plan is supposed to 'break down silos' and 'encourage spontaneous innovation'.","contempt",True,"sarcastic repetition of management clichés"),
("Employee_A","The only thing it's 'breaking down' is my ability to remember what a semicolon is for.","distraction",False,"cognitive load of office noise"),
("Manager","I'm considering 'innovating' a new policy where we all just wear earmuffs and communicate via mime.","joy",True,"sarcastic joy at the absurdity of office life"),
("Employee_A","I've started working in the supply closet because it's the only place with a door that closes.","distraction",True,"extreme measures taken to avoid distraction"),
("Manager","The supply closet is 'off-limits for unauthorized deep-work' — please return to your 'collaborative' desk.","disapproval",True,"sarcastic disapproval of the employee's solution"),
]},
{"scenario":"casual","topic":"serenity_beach","utterances":[
("Traveller","The tide is out and the sand is perfectly flat and reflects the sky like a giant mirror.","serenity",False,"peaceful coastal imagery"),
("Partner","I haven't heard anything except the gulls and the wind for the last two hours.","serenity",False,"auditory peace in nature"),
("Traveller","My mind finally feels like it's stopped racing — for the first time in months.","serenity",False,"mental stillness achieved"),
("Partner","Let's stay here until the sun goes down — there's nowhere else we actually need to be.","trust",False,"shared commitment to the peaceful moment"),
("Traveller","I could live in this specific moment for a very long time and never get tired of it.","serenity",False,"contentment in the present"),
("Partner","It's nice to remember that the world has these pockets of absolute stillness.","acceptance",False,"gratitude for the experience"),
]},
{"scenario":"social","topic":"admiration_craft","utterances":[
("Customer","The table you built has a joinery detail that I've only ever seen in museums before.","admiration",False,"praise for exceptional craftsmanship"),
("Maker","I spent forty hours just on that specific corner to make sure the grain matched perfectly.","vigilance",False,"dedication to detail described"),
("Customer","Most people would have just used a bracket and hidden it — you've made the structure a piece of art.","admiration",False,"praising integrity over convenience"),
("Maker","I believe that if you're going to build something, you should build it to last three hundred years.","trust",False,"sharing a philosophy of quality"),
("Customer","You can feel the care that went into every surface — it's not just a piece of furniture.","admiration",False,"visceral response to quality work"),
("Maker","Thank you. It's rare to have a client who notices the parts that are designed to be invisible.","joy",False,"joy at being understood and appreciated"),
]},
{"scenario":"casual","topic":"terror_storm","utterances":[
("Resident_1","The tornado siren has been going for ten minutes and the sky has turned a bruised shade of green.","terror",False,"imminent severe weather threat"),
("Resident_2","Get the dogs and the emergency kit into the basement right now — don't wait for the wind to start.","vigilance",False,"urgent protective action"),
("Resident_1","I can hear a sound like a freight train coming from the west — is that what it's supposed to sound like.","terror",False,"auditory confirmation of deadly threat"),
("Resident_2","Yes. Close the door, get under the workbench, and stay down until I tell you otherwise.","fear",False,"authoritative direction in a crisis"),
("Resident_1","I'm terrified the house won't hold — I can feel the pressure changing in my ears.","terror",False,"visceral fear of structural failure"),
("Resident_2","We are in the strongest part of the foundation. Stay low and keep your head covered.","trust",False,"reassurance based on safety protocol"),
]},
{"scenario":"conflict","topic":"loathing_abuse","utterances":[
("Victim","He spent ten years convincing me that I was lucky he even tolerated my presence.","loathing",False,"self-directed loathing resulting from abuse"),
("Advocate","That is a classic tactic used to strip a person of their agency and their sense of worth.","trust",False,"professional validation of the victim's experience"),
("Victim","I look at photos from that time and I feel a physical revulsion for the person I allowed myself to become.","loathing",False,"self-loathing and shame"),
("Advocate","The revulsion belongs to his actions — not to your survival. You were doing what you had to do.","acceptance",False,"reframing the shame toward the perpetrator"),
("Victim","I want to erase every memory of him and every trace of that version of myself from the world.","loathing",False,"desire to purge the past"),
("Advocate","We can't erase it, but we can build something so much stronger on top of it that it no longer defines you.","optimism",False,"hopeful path toward recovery"),
]},
{"scenario":"social","topic":"apprehension_speech","utterances":[
("Speaker","There are five hundred people in that room and I'm pretty sure I'm going to vomit on my shoes.","apprehension",False,"intense public speaking anxiety"),
("Organizer","You've practiced this a dozen times and your message is exactly what this audience needs to hear.","trust",False,"encouragement and affirmation of value"),
("Speaker","What if my mind goes blank and I just stand there staring at them in total silence.","apprehension",False,"fear of public failure"),
("Organizer","If that happens, take a sip of water, look at me in the front row, and just read the first line.","trust",False,"practical safety net provided"),
("Speaker","My hands are shaking so much I don't even know if I can hold the microphone steady.","apprehension",False,"visible physical symptoms of fear"),
("Organizer","The audience wants you to succeed. They are on your side. Deep breath — let's go.","serenity",False,"final calming encouragement"),
]},
{"scenario":"workplace","topic":"distraction_notifications","utterances":[
("Developer","I get a notification every time someone comments on a ticket that isn't even in my department.","distraction",False,"excessive and irrelevant digital notifications"),
("IT_Lead","It's part of our 'radical transparency' initiative to keep everyone 'fully informed' at all times.","contempt",True,"sarcastic use of corporate buzzwords"),
("Developer","I'm so 'fully informed' that I haven't had more than ten minutes of uninterrupted focus today.","distraction",False,"negative impact of transparency on productivity"),
("IT_Lead","We're 'optimizing' for 'omniscience' — even if it means no one actually gets any work done.","joy",True,"sarcastic joy at the absurdity"),
("Developer","I'm going to 'transparently' turn off my internet until I've actually finished this feature.","aggressiveness",True,"sarcastic aggressive response to distraction"),
("IT_Lead","That would be an 'unauthorized gap in your organizational awareness' — please remain distracted.","disapproval",True,"sarcastic disapproval of the employee's solution"),
]},
{"scenario":"casual","topic":"serenity_library","utterances":[
("Visitor","The library has that specific smell of old paper and silence that makes me feel like I can finally breathe.","serenity",False,"comforting sensory experience of a library"),
("Librarian","It's one of the few places left where you aren't expected to buy anything or be anywhere else.","trust",False,"affirming the library as a sanctuary"),
("Visitor","I found a corner with a worn leather chair and I've been reading the same page for twenty minutes just because I can.","serenity",False,"enjoying the absence of pressure"),
("Librarian","The world outside is very loud — it's important to have a place where the volume is turned down.","acceptance",False,"shared value of quiet spaces"),
("Visitor","I feel like my brain is finally untangling itself after a week of total chaos.","serenity",False,"mental recovery through stillness"),
("Librarian","Stay as long as you need — we don't close for another four hours.","serenity",False,"unconditional hospitality"),
]},
{"scenario":"social","topic":"admiration_teacher","utterances":[
("Parent","My son hasn't stopped talking about the science project you helped him start last week.","admiration",False,"praise for inspiring a child"),
("Teacher","He has a natural curiosity that just needs the right spark to turn into a real passion.","trust",False,"modest acceptance and focus on the student"),
("Parent","The way you managed to explain quantum physics to a ten-year-old was nothing short of miraculous.","admiration",False,"praising exceptional communication skill"),
("Teacher","It's about finding the story in the data — if they care about the story, they'll learn the math.","interest",False,"sharing a teaching philosophy"),
("Parent","You've changed how he sees the world and how he sees himself as a student — thank you.","admiration",False,"recognizing profound positive impact"),
("Teacher","That is exactly why I'm in this profession — it's an honor to be part of his journey.","joy",False,"genuine fulfillment in work"),
]},
{"scenario":"casual","topic":"terror_wilderness","utterances":[
("Hiker_1","I just saw a mountain lion on the trail about fifty yards ahead and it's looking directly at us.","terror",False,"encounter with dangerous predator"),
("Hiker_2","Do not run. Stand tall, make yourself as big as possible, and start making a lot of noise.","vigilance",False,"crisis management protocol"),
("Hiker_1","It's starting to move toward us — oh my god, it's actually coming this way.","terror",False,"escalation of life-threatening threat"),
("Hiker_2","Keep shouting! Raise your poles! Do not turn your back on it for even a second.","fear",False,"urgent defensive instructions"),
("Hiker_1","I'm paralyzed — I can't even get my breath out to scream.","terror",False,"freezing response to extreme fear"),
("Hiker_2","I'm right here. We are doing this together. Keep your eyes on it and keep backing away slowly.","trust",False,"steadfast support in a life-or-death situation"),
]},
{"scenario":"conflict","topic":"loathing_betrayal_v2","utterances":[
("Friend_A","I can't believe you told the whole office about my private medical situation just to get a laugh.","loathing",False,"betrayal of deep personal trust"),
("Friend_B","I didn't think it was that big of a deal — everyone was just joking around and I joined in.","contempt",True,"sarcastic and dismissive defense of betrayal"),
("Friend_A","You used my most vulnerable moment as social currency to make yourself look interesting.","loathing",False,"revulsion at friend's opportunistic behavior"),
("Friend_B","You're being oversensitive — it's not like I told them something that wasn't already true.","disapproval",False,"victim-blaming and lack of remorse"),
("Friend_A","The fact that you don't even see why this is a violation is exactly why we are finished.","loathing",False,"final loss of respect and end of friendship"),
("Friend_B","Fine, have your drama — I'm sure you'll enjoy being the martyr for a week.","contempt",True,"sarcastic contempt for the other's pain"),
]},
{"scenario":"social","topic":"apprehension_audition","utterances":[
("Actor","I've wanted this role for three years and my entire future feels like it's hanging on the next five minutes.","apprehension",False,"high-stakes career anxiety"),
("Agent","You are the best prepared person in that waiting room — just go in there and show them what you've built.","trust",False,"professional encouragement"),
("Actor","I can feel my throat tightening up — what if I can't even get the first line of the monologue out.","apprehension",False,"physical symptoms of performance anxiety"),
("Agent","If that happens, use it. The character is supposed to be vulnerable — let the fear work for you.","trust",False,"reframing anxiety as a professional tool"),
("Actor","I'm going to be sick — I genuinely think I'm going to be physically ill in front of the director.","apprehension",False,"extreme physiological response to stress"),
("Agent","Focus on the work, not the outcome. You are an actor — go act.","serenity",False,"grounding advice focused on the present"),
]},
{"scenario":"workplace","topic":"distraction_meetings_v2","utterances":[
("Analyst","I have six hours of 'mandatory brainstorming' today and forty minutes to actually do my job.","distraction",False,"excessive meetings preventing actual work"),
("Manager","We're 'leveraging collective intelligence' to 'ideate on our multi-year strategic roadmap'.","contempt",True,"sarcastic use of corporate jargon"),
("Analyst","The only 'intelligence' being 'leveraged' is my ability to look awake while my brain is in standby mode.","distraction",False,"mental exhaustion from pointless meetings"),
("Manager","We're 'optimizing' for 'alignment' — even if the 'alignment' is just everyone being equally frustrated.","joy",True,"sarcastic joy at the inefficiency"),
("Analyst","I'm going to 'brainstorm' a way to be in two places at once — or just one place that isn't this room.","aggressiveness",True,"sarcastic aggressive response"),
("Manager","That would be a 'unilateral departure from our collaborative culture' — please remain ideating.","disapproval",True,"sarcastic disapproval of the analyst's desire to work"),
]},
{"scenario":"casual","topic":"serenity_snow","utterances":[
("Resident_1","The snow is falling so thickly that I can't even see the houses across the street anymore.","serenity",False,"peaceful winter imagery"),
("Resident_2","It's that total silence that only happens when the world is covered in a foot of fresh powder.","serenity",False,"auditory peace of a snowstorm"),
("Resident_1","The fireplace is going and I have a book and there is absolutely nowhere we have to go today.","serenity",False,"contentment and safety in a storm"),
("Resident_2","I can feel the entire week of stress just evaporating as I watch the flakes drift down.","serenity",False,"stress relief through nature connection"),
("Resident_1","It's a perfect day to just exist without any expectation of being productive.","serenity",False,"permission to be still"),
("Resident_2","Let's make some tea and just sit by the window for a while — the world can wait.","acceptance",False,"shared acceptance of the quiet moment"),
]},
{"scenario":"social","topic":"admiration_courage","utterances":[
("Witness","He ran back into the building twice to make sure everyone from the second floor was out safely.","admiration",False,"praising physical courage and selflessness"),
("Reporter","He isn't a firefighter — he was just a bystander who happened to be walking past when it happened.","amazement",False,"magnitude of spontaneous bravery recognized"),
("Witness","The level of calm he displayed while the roof was literally starting to sag was extraordinary.","admiration",False,"praising composure in a crisis"),
("Reporter","He says he didn't even think about it — he just saw people who needed help and moved.","trust",False,"hero's modest motivation described"),
("Witness","That kind of instinctive heroism is what keeps the world from falling apart, I think.","admiration",False,"heroism recognized as a fundamental social good"),
("Reporter","I'm going to make sure his name is at the very top of the story — he deserves the recognition.","joy",False,"joy at being able to honor a hero"),
]},
{"scenario":"casual","topic":"terror_fire","utterances":[
("Tenant","I woke up to the sound of the smoke alarm and the hallway was already filled with thick black smoke.","terror",False,"midnight fire in a residential building"),
("Dispatcher","Stay low to the ground where the air is clearer and move toward the nearest fire exit immediately.","vigilance",False,"emergency life-saving instructions"),
("Tenant","The door handle is hot — oh god, I can't get out through the hallway, the fire is right outside.","terror",False,"trapped by fire with no obvious escape"),
("Dispatcher","Do not open the door. Go to the window, hang a sheet out so we can see you, and stay as low as possible.","fear",False,"urgent redirection in a life-threatening crisis"),
("Tenant","I can hear the fire roaring on the other side of the wall — please hurry, I'm so scared.","terror",False,"visceral fear of imminent death"),
("Dispatcher","The trucks are arriving right now. I am staying on the line with you until they reach your window.","trust",False,"steadfast support and commitment to safety"),
]},
{"scenario":"conflict","topic":"loathing_exploitation","utterances":[
("Employee","I found out the company has been systematically underpaying the overseas staff for three years.","loathing",False,"revulsion at corporate exploitation"),
("Executive","We are in full compliance with the local labor laws of every jurisdiction where we operate.","contempt",True,"sarcastic and legalistic defense of exploitation"),
("Employee","Using legal loopholes to pay people a fraction of their worth is a moral failure, not a legal one.","loathing",False,"moral revulsion at corporate behavior"),
("Executive","If you find our 'global compensation strategy' so offensive, you are free to seek employment elsewhere.","disapproval",False,"defensive and dismissive reaction to criticism"),
("Employee","I will be seeking employment elsewhere — and taking the full documentation of this 'strategy' to the press.","loathing",False,"sustained revulsion and commitment to accountability"),
("Executive","Good luck with that — I'm sure the 'press' will be fascinated by your 'brave' discovery of capitalism.","contempt",True,"sarcastic contempt for the employee's ethics"),
]},
{"scenario":"social","topic":"apprehension_proposal","utterances":[
("Partner_A","I have the ring in my pocket and I'm pretty sure I'm about to pass out from sheer terror.","apprehension",False,"intense proposal anxiety"),
("Friend","She loves you, she's been dropping hints for months, and there is no way she's going to say no.","trust",False,"reassurance of a positive outcome"),
("Partner_A","But what if I trip, or forget what I was going to say, or the ring falls into the fountain.","apprehension",False,"fear of public embarrassment and failure"),
("Friend","If any of that happens, it just becomes a better story to tell at the wedding.","optimism",False,"reframing potential failure as future joy"),
("Partner_A","I can't even remember my own middle name right now — how am I supposed to give a speech.","apprehension",False,"cognitive load of extreme anxiety"),
("Friend","Just look her in the eyes and tell her why you want to spend the rest of your life with her. The rest is just noise.","serenity",False,"grounding advice focused on the core emotion"),
]},
{"scenario":"workplace","topic":"distraction_coworkers","utterances":[
("Developer","I'm trying to debug a complex race condition and my neighbor is having a loud debate about pizza.","distraction",False,"irrelevant and loud office conversation breaking focus"),
("Team_Lead","The 'open-office experience' is a 'dynamic tapestry of diverse perspectives'.","contempt",True,"sarcastic use of corporate buzzwords"),
("Developer","The 'perspective' I need right now is the one where I don't want to throw my monitor at the wall.","distraction",False,"frustration and loss of focus due to noise"),
("Team_Lead","We're 'optimizing' for 'serendipitous collisions' — like colliding with a deadline you're about to miss.","joy",True,"sarcastic joy at the inefficiency"),
("Developer","I'm going to 'serendipitously collide' with the 'do not disturb' mode on my headphones for the rest of the day.","aggressiveness",True,"sarcastic aggressive response to distraction"),
("Team_Lead","That would be an 'unauthorized withdrawal from our collective consciousness' — please remain available for pizza debates.","disapproval",True,"sarcastic disapproval of the developer's solution"),
]},
{"scenario":"casual","topic":"serenity_forest","utterances":[
("Hiker","The deeper I get into the woods, the more the world outside feels like a distant, irrelevant dream.","serenity",False,"peaceful isolation in nature"),
("Partner","There's no cell signal here, which is probably the most therapeutic thing I've experienced all year.","serenity",False,"relief at digital disconnection"),
("Hiker","The only sounds are the wind in the pines and the occasional bird — and our own footsteps.","serenity",False,"auditory peace in the forest"),
("Partner","I can feel my shoulders finally dropping away from my ears for the first time in weeks.","serenity",False,"physical relief through nature connection"),
("Hiker","Let's stop at the creek and just listen to the water for a while — there's no rush.","acceptance",False,"shared commitment to the present moment"),
("Partner","I think this is exactly what my brain needed to stop vibrating from all the noise.","serenity",False,"mental recovery through stillness"),
]},
{"scenario":"social","topic":"admiration_doctor","utterances":[
("Patient","You spent three hours explaining the procedure to my daughter until she felt safe enough to go in.","admiration",False,"praising exceptional patient care"),
("Doctor","Medicine isn't just about the surgery — it's about making sure the person is ready for it.","trust",False,"sharing a professional philosophy of care"),
("Patient","Most doctors would have just given us a pamphlet and moved on to the next appointment.","admiration",False,"praising integrity over convenience"),
("Doctor","I became a doctor to help people, not to process insurance codes as quickly as possible.","admiration",False,"selfless motivation recognized"),
("Patient","You've made a terrifying day feel manageable and even hopeful — thank you for that.","admiration",False,"recognizing profound positive impact"),
("Doctor","It's an honor to be the one you trust with her care — I'll see you both in the recovery room.","joy",False,"genuine fulfillment in helping others"),
]},
{"scenario":"casual","topic":"terror_car_accident","utterances":[
("Driver","I saw the other car running the red light and I knew I couldn't stop in time — the impact was deafening.","terror",False,"witnessing and experiencing a high-speed crash"),
("Witness","The car is completely crushed — oh my god, are you able to move your legs at all.","apprehension",False,"urgent concern for life-altering injury"),
("Driver","I can't get the door open and I can smell smoke — please help me get out of here.","terror",False,"trapped in a burning vehicle after a crash"),
("Witness","The fire department is coming! I'm going to try to break the window from the outside — stay back!","vigilance",False,"urgent rescue attempt"),
("Driver","I'm so scared it's going to explode — please don't leave me in here.","terror",False,"visceral fear of imminent death"),
("Witness","I am not going anywhere. I have the window open — give me your hand and I'll pull you through.","trust",False,"steadfast support and rescue in a crisis"),
]},
{"scenario":"conflict","topic":"loathing_plagiarism","utterances":[
("Student_A","I found my entire thesis published under your name in the department's annual journal.","loathing",False,"revulsion at intellectual theft and betrayal"),
("Student_B","I just used your 'draft' as a 'foundation' for my own 'expanded research'.","contempt",True,"sarcastic and deceptive defense of plagiarism"),
("Student_A","You copied my data, my analysis, and even my specific phrasing word-for-word.","loathing",False,"moral revulsion at the theft"),
("Student_B","You should be honored that your 'ideas' were 'sufficiently developed' to be included in my work.","disapproval",False,"defensive and dismissive reaction to the accusation"),
("Student_A","I look at you and I see a parasite who lacks even the basic dignity to do their own work.","loathing",False,"fundamental loss of respect and revulsion"),
("Student_B","Have fun explaining your 'theory' to the ethics committee — I'm sure they'll be fascinated by your 'contribution'.","contempt",True,"sarcastic contempt for the victim's attempts at justice"),
]},
{"scenario":"social","topic":"apprehension_interview","utterances":[
("Applicant","I'm sitting in the lobby and I've forgotten how to explain what my own job title actually means.","apprehension",False,"intense interview anxiety"),
("Friend","You are the most qualified person they've seen all week — just go in there and tell them the truth.","trust",False,"encouragement and affirmation of worth"),
("Applicant","My heart is beating so hard I'm worried they'll be able to hear it through my shirt.","apprehension",False,"physical symptoms of high-stakes anxiety"),
("Friend","If they can hear it, it just shows how much you actually care about the role.","optimism",False,"reframing anxiety as passion"),
("Applicant","I can feel a cold sweat starting and I'm pretty sure I'm going to forget my own name.","apprehension",False,"physiological response to extreme stress"),
("Friend","You are prepared, you are capable, and you belong in that room. Deep breath — you've got this.","serenity",False,"final grounding and calming encouragement"),
]},
{"scenario":"workplace","topic":"distraction_email","utterances":[
("Analyst","I've received forty emails since nine AM and none of them require my actual input.","distraction",False,"excessive and irrelevant digital communication"),
("Manager","We're 'optimizing' for 'asynchronous alignment' to ensure 'total organizational visibility'.","contempt",True,"sarcastic use of corporate buzzwords"),
("Analyst","The only thing I'm 'aligned' with is the 'delete' button — it's the most productive tool I have.","distraction",False,"negative impact of digital noise on work"),
("Manager","We're 'leveraging' the 'power of the inbox' to 'socialize our core competencies'.","joy",True,"sarcastic joy at the inefficiency"),
("Analyst","I'm going to 'socialize' my inbox by 'archiving' every message until I've actually done some work.","aggressiveness",True,"sarcastic aggressive response to distraction"),
("Manager","That would be a 'unilateral departure from our transparent workflow' — please remain overwhelmed.","disapproval",True,"sarcastic disapproval of the analyst's desire to focus"),
]},
{"scenario":"casual","topic":"serenity_mountains","utterances":[
("Traveller","The air up here is so thin and cold that it feels like it's scrubbing my lungs clean.","serenity",False,"peaceful mountain imagery"),
("Partner","We haven't seen another person for three days and I've completely forgotten what a car sounds like.","serenity",False,"relief at isolation from modern noise"),
("Traveller","The stars at night are so bright they actually cast a shadow on the ground — it's extraordinary.","serenity",False,"wonder and peace in nature"),
("Partner","I can feel the entire year of city noise just falling away as we walk.","serenity",False,"mental recovery through nature connection"),
("Traveller","Let's stop here and just watch the light change on the peaks for the afternoon.","serenity",False,"shared commitment to the quiet moment"),
("Partner","It's a perfect place to just exist without any need to be anything for anyone.","acceptance",False,"gratitude for the stillness"),
]},
{"scenario":"social","topic":"admiration_caregiver","utterances":[
("Son","You've been taking care of Dad for twelve years and I've never once heard you complain about the cost.","admiration",False,"praising long-term selfless caregiving"),
("Mother","He's the love of my life — there is nowhere else I would rather be than right here with him.","trust",False,"modest and dedicated motivation"),
("Son","The level of patience you have when he's having a bad day is something I genuinely can't comprehend.","admiration",False,"praising exceptional emotional resilience"),
("Mother","Patience is just another way of saying I remember the person he was and the person he still is inside.","love",False,"sharing a philosophy of compassionate care"),
("Son","You're the strongest person I know, Mom — I don't think I've ever told you that properly.","admiration",False,"recognizing profound personal strength"),
("Mother","Thank you, honey. Knowing you see it and you're here with me makes it a lot easier to keep going.","joy",False,"joy at being seen and supported"),
]},
{"scenario":"casual","topic":"terror_earthquake","utterances":[
("Resident_1","The floor just suddenly turned into a liquid and the sound of the building groaning was terrifying.","terror",False,"experiencing a major earthquake"),
("Resident_2","Get under the table and hold onto the legs — don't try to run for the door until it stops.","vigilance",False,"earthquake safety protocol"),
("Resident_1","A bookshelf just collapsed and the power went out — oh my god, I can hear the ceiling cracking.","terror",False,"escalating structural threat in a crisis"),
("Resident_2","Stay down and protect your head! I'm right here with you — just hold on!","fear",False,"urgent protection and companionship in danger"),
("Resident_1","I'm terrified the whole building is going to come down on top of us — the shaking won't stop.","terror",False,"visceral fear of structural collapse"),
("Resident_2","The shaking is slowing down. Stay exactly where you are until I can check the exit.","trust",False,"reassurance based on safety protocol"),
]},
{"scenario":"conflict","topic":"loathing_deception","utterances":[
("Employee","I found out you've been using my personal login to authorize payments to your own 'consulting' firm.","loathing",False,"revulsion at financial fraud and betrayal"),
("Manager","I was just 'optimizing' our 'vendor-relationship infrastructure' to 'accelerate our growth'.","contempt",True,"sarcastic and deceptive defense of fraud"),
("Employee","You used my identity to commit a crime and you're trying to frame it as a 'growth strategy'.","loathing",False,"moral revulsion at the manager's actions"),
("Manager","You should be 'grateful' for the 'opportunity' to be part of such a 'dynamic' financial model.","disapproval",False,"defensive and dismissive reaction to the accusation"),
("Employee","I look at you and I see a common thief who lacks the courage to even admit what they've done.","loathing",False,"fundamental loss of respect and revulsion"),
("Manager","Good luck 'reporting' me — I'm sure the 'authorities' will be fascinated by your 'unauthorized access' to my 'strategy'.","contempt",True,"sarcastic contempt for the employee's attempt at justice"),
]},
{"scenario":"social","topic":"apprehension_competition","utterances":[
("Athlete","I've trained for four years for this one race and my legs feel like they've turned into lead.","apprehension",False,"intense pre-competition anxiety"),
("Coach","That's just your body preparing for the effort — you are faster and stronger than you've ever been.","trust",False,"encouragement and affirmation of physical readiness"),
("Athlete","What if I trip at the start or lose my rhythm in the first hundred meters.","apprehension",False,"fear of failure in a high-stakes moment"),
("Coach","If that happens, you adjust and you keep moving. You've practiced every scenario a thousand times.","trust",False,"reassurance based on rigorous preparation"),
("Athlete","I can feel my pulse in my teeth — I'm pretty sure I'm about to faint before the gun goes off.","apprehension",False,"physiological symptoms of extreme performance anxiety"),
("Coach","Focus on the first ten meters. Nothing else exists right now. Deep breath — let's go.","serenity",False,"grounding advice focused on the immediate task"),
]},
{"scenario":"workplace","topic":"distraction_chatter","utterances":[
("Developer","I'm trying to solve a recursive bug and my neighbor is describing every detail of their weekend hike.","distraction",False,"irrelevant and loud office chatter breaking focus"),
("Manager","The 'open-office environment' is a 'vibrant marketplace of social capital'.","contempt",True,"sarcastic use of corporate buzzwords"),
("Developer","The only 'capital' I need right now is the one that lets me buy a room with a lock on it.","distraction",False,"frustration and loss of focus due to noise"),
("Manager","We're 'optimizing' for 'organic relationship-building' — even if it's at the expense of our 'release schedule'.","joy",True,"sarcastic joy at the inefficiency"),
("Developer","I'm going to 'organically build' a relationship with the 'mute' button on my headphones for the afternoon.","aggressiveness",True,"sarcastic aggressive response to distraction"),
("Manager","That would be an 'unauthorized withdrawal from our collective energy' — please remain vibrant.","disapproval",True,"sarcastic disapproval of the developer's solution"),
]},
{"scenario":"casual","topic":"serenity_rain","utterances":[
("Resident_1","The rain is coming down so steadily that it's turned the whole world into a soft grey blur.","serenity",False,"peaceful rainy imagery"),
("Resident_2","It's the perfect sound to just listen to while you're curled up on the couch with a hot tea.","serenity",False,"auditory peace of a rainstorm"),
("Resident_1","I've been staring at the window for an hour and I haven't felt the need to check my phone even once.","serenity",False,"digital detox and peace through nature"),
("Resident_2","I can feel the entire week of pressure just slowly leaking out of my brain as I listen.","serenity",False,"stress relief through stillness"),
("Resident_1","It's nice to have a day where the weather gives you permission to just do absolutely nothing.","serenity",False,"contentment in the absence of pressure"),
("Resident_2","Let's just stay here and listen to the world wash itself clean for a while.","acceptance",False,"shared gratitude for the quiet moment"),
]},
{"scenario":"social","topic":"admiration_leader","utterances":[
("Member","The way you handled that crisis in the community meeting was a masterclass in diplomacy and grace.","admiration",False,"praising exceptional leadership and composure"),
("Leader","I just wanted to make sure everyone felt heard before we tried to find a solution together.","trust",False,"modest and inclusive motivation"),
("Member","Most people would have started shouting back the moment the accusations started flying.","admiration",False,"praising integrity over reactive anger"),
("Leader","Shouting doesn't solve problems — it just makes the people who are already hurting feel even more unheard.","interest",False,"sharing a philosophy of compassionate leadership"),
("Member","You're the heart of this neighborhood — I don't know where we'd be without your steady hand.","admiration",False,"recognizing profound positive impact"),
("Leader","Thank you. Knowing I have your support makes the harder days feel a lot more worth the effort.","joy",False,"genuine fulfillment in community work"),
]},
{"scenario":"casual","topic":"terror_attack","utterances":[
("Witness","The explosion was so loud I could feel the shockwave in my teeth and the glass just started raining down.","terror",False,"witnessing a major public tragedy"),
("Survivor","I'm covered in dust and I can't find my friend — oh god, everyone is running and I don't know where to go.","terror",False,"panic and loss in a crisis"),
("Witness","Stay away from the buildings! Follow me toward the park — it's the most open space nearby.","vigilance",False,"urgent direction toward safety"),
("Survivor","I think I'm hit — there's blood on my sleeve and I can't feel my arm properly.","terror",False,"trauma and physical shock"),
("Witness","I've got you. Just keep moving toward the light and don't look back. We are almost there.","trust",False,"steadfast support and rescue in a crisis"),
("Survivor","I've never seen anything like this — I thought the world was ending.","terror",False,"encounter with extreme public violence"),
]},
{"scenario":"conflict","topic":"loathing_neglect","utterances":[
("Employee","I discovered that the safety protocols on the production line have been bypassed for six months.","loathing",False,"revulsion at corporate disregard for human safety"),
("Manager","We are 'streamlining our operational efficiency' to 'maximize our shareholder value'.","contempt",True,"sarcastic and legalistic defense of neglect"),
("Employee","You are gambling with people's lives to save a few dollars on the balance sheet and it's disgusting.","loathing",False,"moral revulsion at corporate behavior"),
("Manager","If you find our 'operational strategy' so offensive, you are free to seek employment in a less 'efficient' company.","disapproval",False,"defensive and dismissive reaction to criticism"),
("Employee","I will be seeking employment elsewhere — and taking the full documentation of this 'strategy' to the safety board.","loathing",False,"sustained revulsion and commitment to accountability"),
("Manager","I'm sure the 'safety board' will be fascinated by your 'unauthorized investigation' into our 'success'.","contempt",True,"sarcastic contempt for the employee's ethics"),
]},
{"scenario":"social","topic":"apprehension_outcome","utterances":[
("Patient","The doctor is walking down the hallway with the test results and my heart is about to beat out of my chest.","apprehension",False,"intense medical anxiety"),
("Partner","Whatever it says, we handle it together. We are a team and we have a plan for every outcome.","trust",False,"reassurance of companionship regardless of result"),
("Patient","I can feel a cold sweat and I'm pretty sure I'm about to faint before he even reaches the door.","apprehension",False,"physiological symptoms of life-altering news"),
("Partner","Deep breaths — four counts in, four counts out. Focus on my hand, not the hallway.","serenity",False,"grounding advice focused on the present"),
("Patient","If it's bad news, I don't know how I'm going to tell the kids — I'm so scared of their reaction.","apprehension",False,"family-linked anxiety"),
("Partner","We will tell them together, when the time is right. Right now, just breathe.","trust",False,"steadfast support in a high-stakes moment"),
]},
{"scenario":"workplace","topic":"distraction_interruptions","utterances":[
("Designer","I've had six 'quick questions' in the last hour and none of them were actually quick.","distraction",False,"frequent 'quick' interruptions breaking deep focus"),
("Manager","We're 'fostering a culture of immediate feedback' to 'accelerate our collaborative loops'.","contempt",True,"sarcastic use of corporate buzzwords"),
("Designer","The only 'loop' I'm in right now is the one where I keep forgetting what I was working on.","distraction",False,"negative impact of frequent context switching"),
("Manager","We're 'optimizing' for 'real-time alignment' — even if it means no one actually has a 'real-time' to work.","joy",True,"sarcastic joy at the inefficiency"),
("Designer","I'm going to 'align' my door with a 'closed' position for the rest of the afternoon.","aggressiveness",True,"sarcastic aggressive response to distraction"),
("Manager","That would be an 'unauthorized barrier to our transparent communication' — please remain interruptible.","disapproval",True,"sarcastic disapproval of the designer's solution"),
]},
{"scenario":"casual","topic":"surprise_lottery","utterances":[
("Winner","I just checked my ticket and I actually won the ten million dollar jackpot.","surprise",False,"initial shock of winning lottery"),
("Friend","You are joking — show me the ticket right now, there's no way that's real.","amazement",False,"disbelief and amazement"),
("Winner","Look at the numbers — they match every single one on the screen.","surprise",False,"confirming the impossible win"),
("Friend","Oh my god, your life is never going to be the same after this moment.","joy",False,"joy for friend's fortune"),
("Winner","I can't even stand up — my legs have turned into water.","surprise",False,"physical shock response"),
("Friend","Sit down before you fall down — I'm calling your parents right now!","vigilance",False,"taking charge in a moment of shock"),
]},
{"scenario":"casual","topic":"anticipation_concert","utterances":[
("Fan","The lights just went down and the intro music is starting — they're about to come on stage.","anticipation",False,"excitement at start of concert"),
("Friend","I've been waiting for this tour for three years and I can't believe we're finally here.","ecstasy",False,"intense joy and fulfillment"),
("Fan","I can see the silhouettes of the band moving behind the curtain!","anticipation",False,"heightened expectation"),
("Friend","The crowd is going absolutely wild — I can't even hear myself scream.","ecstasy",False,"intense collective joy"),
("Fan","Here they come! Look at the pyrotechnics!","anticipation",False,"climax of anticipation"),
("Friend","This is the best night of my entire life!","ecstasy",False,"peak emotional experience"),
]},
{"scenario":"social","topic":"fear_shadow","utterances":[
("Partner_A","I just saw a shadow move across the backyard and the security light didn't turn on.","fear",False,"suspicion of intruder"),
("Partner_B","It was probably just a deer or a neighbor's cat — don't overthink it.","trust",False,"attempted reassurance"),
("Partner_A","No, it was too tall for an animal — and it's standing perfectly still near the shed.","fear",False,"heightened threat perception"),
("Partner_B","I'm going to get the flashlight — stay here and keep the door locked.","vigilance",False,"taking defensive action"),
("Partner_A","I can hear footsteps on the gravel now — oh god, they're coming toward the porch.","fear",False,"imminent threat of intrusion"),
("Partner_B","I'm calling the police right now — stay away from the windows.","vigilance",False,"emergency response"),
]},
{"scenario":"workplace","topic":"anger_mistake","utterances":[
("Manager","I just found out you deleted the entire client database because you didn't check the environment.","anger",False,"manager's fury over major error"),
("Employee","I thought I was on the staging server — the labels are almost identical.","fear",False,"anxious explanation for mistake"),
("Manager","Thinking is not enough when you're dealing with five years of proprietary data!","anger",False,"escalated anger and frustration"),
("Employee","I'm already trying to run the restore from the backup we made this morning.","vigilance",False,"focus on recovery"),
("Manager","If that backup fails, you don't even need to bother coming in tomorrow.","anger",False,"threat of termination"),
("Employee","I understand the severity — I'm not leaving until this is fixed.","acceptance",False,"taking responsibility"),
]},
{"scenario":"casual","topic":"surprise_proposal","utterances":[
("Woman","He took me to the top of the hill and suddenly he was down on one knee with a ring.","surprise",False,"shock of unexpected marriage proposal"),
("Best_Friend","I knew he was going to do it today! I've been hiding the secret for weeks!","joy",False,"vicarious joy"),
("Woman","I was so stunned I forgot how to say the word 'yes' for a full thirty seconds.","surprise",False,"cognitive freeze from shock"),
("Best_Friend","The photos he sent me are incredible — you look like you've seen a ghost.","joy",False,"laughing at friend's reaction"),
("Woman","I still haven't stopped shaking — it feels like a dream.","surprise",False,"lingering sense of unreality"),
("Best_Friend","We are going to start planning the engagement party tomorrow!","anticipation",False,"future-focused excitement"),
]},
{"scenario":"social","topic":"anticipation_birth","utterances":[
("Father","The doctor says she's at ten centimeters — we're going to meet him in the next hour.","anticipation",False,"excitement and anxiety of imminent birth"),
("Grandmother","I've been waiting twenty years for this moment — I'm so proud of both of you.","love",False,"affection and pride"),
("Father","I can't stop pacing — I feel like I have more energy than I know what to do with.","anticipation",False,"physical manifestation of excitement"),
("Grandmother","That's just the fatherly instinct kicking in — you're ready for this.","trust",False,"reassurance"),
("Father","I can hear him crying! He's here! They just brought him out!","ecstasy",False,"intense joy at birth"),
("Grandmother","Welcome to the world, little one. We've all been waiting for you.","joy",False,"celebration of new life"),
]},
{"scenario":"casual","topic":"fear_loss","utterances":[
("Parent","I turned around for ten seconds in the mall and my four-year-old was gone.","fear",False,"panic of losing a child in public"),
("Security","When was the last time you saw him and what was he wearing today.","vigilance",False,"systematic search initiation"),
("Parent","He was right here by the fountain — he's wearing a bright red t-shirt and blue jeans.","fear",False,"desperate details provided"),
("Security","I'm locking down the exits and putting out a description over the radio right now.","vigilance",False,"taking control of the situation"),
("Parent","What if someone took him — oh god, I can't breathe, I have to find him.","fear",False,"worst-case scenario thinking"),
("Security","We have eyes on him on the cameras — he's just sitting in the toy store window.","serenity",False,"relieving the fear with facts"),
]},
{"scenario":"workplace","topic":"anger_betrayal","utterances":[
("Colleague_A","I just saw your name on the patent for the algorithm that I spent six months developing.","anger",False,"outrage over intellectual property theft"),
("Colleague_B","I made several significant 'contributions' that made it viable for commercial use.","contempt",True,"sarcastic and deceptive justification"),
("Colleague_A","You changed two lines of code and then submitted it behind my back!","anger",False,"specific accusation of betrayal"),
("Colleague_B","In this company, 'viability' is more important than 'authorship' — you should learn that.","contempt",True,"sarcastic and dismissive corporate logic"),
("Colleague_A","I'm going directly to HR with the original git history and my lab notes.","anger",False,"committing to a formal complaint"),
("Colleague_B","I'm sure they'll be 'very impressed' with your 'detective work' — good luck.","contempt",True,"sarcastic contempt for the consequences"),
]},
{"scenario":"casual","topic":"surprise_return","utterances":[
("Soldier","I walked into the kitchen while she was making dinner and she dropped the plate.","surprise",False,"shock of unexpected return from deployment"),
("Wife","I thought you were still three thousand miles away and I wasn't going to see you for months.","surprise",False,"disbelief and amazement"),
("Soldier","The mission ended early and I wanted to make sure it was a total surprise.","joy",False,"happiness at successful surprise"),
("Wife","I'm still convinced I'm dreaming — I don't want to let go of you in case you disappear.","surprise",False,"fear of the surprise being unreal"),
("Soldier","I'm right here. I'm home. And I'm not going back for a very long time.","serenity",False,"comfort and stability"),
("Wife","This is the best surprise I've ever had in my entire life.","joy",False,"peak happiness"),
]},
{"scenario":"social","topic":"anticipation_vacation","utterances":[
("Friend_A","I'm finishing my last email and then I'm turning off my laptop for two full weeks.","anticipation",False,"excitement for upcoming travel"),
("Friend_B","The plane leaves in six hours — have you finished packing your bags yet.","vigilance",False,"checking on preparation"),
("Friend_A","I've been packed for three days — I've even checked the weather at the destination ten times.","anticipation",False,"over-preparation due to excitement"),
("Friend_B","I've already downloaded all the maps and booked the reservations for the first three nights.","vigilance",False,"sharing the preparation load"),
("Friend_A","I can already feel the tropical air and hear the sound of the ocean in my head.","anticipation",False,"vividly imagining the reward"),
("Friend_B","We are going to have the most incredible time — we've earned this.","optimism",False,"positive outlook on the trip"),
]},
{"scenario":"casual","topic":"fear_accident","utterances":[
("Child","I was playing with the ball and it went into the street — I didn't see the car coming.","fear",False,"frightened child after near-miss"),
("Parent","I saw you disappear between the parked cars and my heart literally stopped beating.","fear",False,"parent's intense fear for child's safety"),
("Child","The car made a really loud noise and the man looked really angry at me.","fear",False,"residual fear of the driver's reaction"),
("Parent","The driver was scared too, honey — that's why he was shouting. He didn't want to hit you.","trust",False,"explaining the situation to calm the child"),
("Child","I'm never going to run after the ball again — I promise.","submission",False,"compliance out of fear"),
("Parent","I know you won't. Let's go inside and sit down for a while — we're both a bit shaky.","serenity",False,"calming the aftermath"),
]},
{"scenario":"workplace","topic":"anger_deadline","utterances":[
("Director","The client just called and they're moving the launch date to tomorrow morning.","anger",False,"fury at unreasonable client demand"),
("Lead_Dev","That's physically impossible — we still have twelve major bugs in the tracking system.","fear",False,"panic at impossible workload"),
("Director","I don't care about 'impossible' — I care about the five-million-dollar contract we're about to lose!","anger",False,"escalated pressure and anger"),
("Lead_Dev","If we ship it now, the entire system will crash within twenty minutes of going live.","vigilance",False,"warning of catastrophic failure"),
("Director","Then you have twenty hours to make sure it doesn't — get everyone back in the office right now!","anger",False,"aggressive and unreasonable command"),
("Lead_Dev","I'll call the team, but I'm putting it on record that I advised against this.","acceptance",False,"reluctant compliance"),
]},
{"scenario":"casual","topic":"surprise_gift","utterances":[
("Recipient","I opened the box thinking it was a new toaster and it was actually keys to a car.","surprise",False,"shock of extravagant and unexpected gift"),
("Giver","You've been driving that old rust-bucket for ten years — it was time for an upgrade.","joy",False,"satisfaction in giving"),
("Recipient","I can't believe you did this — I don't even know what to say.","surprise",False,"speechless from shock"),
("Giver","Don't say anything — just go out there and take it for a test drive!","joy",False,"encouraging enjoyment"),
("Recipient","I'm actually crying — I've never had something this nice in my life.","surprise",False,"emotional release from shock"),
("Giver","You deserve every bit of it. Merry Christmas!","love",False,"affectionate closing"),
]},
{"scenario":"social","topic":"anticipation_game","utterances":[
("Fan_1","It's the bottom of the ninth, bases are loaded, and our best hitter is coming to the plate.","anticipation",False,"extreme tension and excitement at end of game"),
("Fan_2","If he hits a home run here, we win the championship for the first time in fifty years.","anticipation",False,"high-stakes expectation"),
("Fan_1","I can't even look — I'm watching through my fingers.","fear",False,"anxiety during high-stakes moment"),
("Fan_2","He just connected! That ball is going, going... it's over the wall!","ecstasy",False,"intense joy at victory"),
("Fan_1","They did it! Oh my god, they actually did it!","ecstasy",False,"collective celebration"),
("Fan_2","I'm going to be celebrating this for the rest of the year!","joy",False,"sustained happiness"),
]},
{"scenario":"casual","topic":"fear_health","utterances":[
("Patient","The doctor found a lump and she wants to do a biopsy immediately to rule out anything serious.","fear",False,"intense anxiety over potential health crisis"),
("Friend","Biopsies are very common — it's just a precaution to make sure everything is okay.","trust",False,"supportive reassurance"),
("Patient","But the way she looked at me when she said it — I've never felt so vulnerable in my life.","fear",False,"fear triggered by non-verbal cues"),
("Friend","I'm going to go with you to the appointment. You don't have to wait for the results alone.","trust",False,"practical commitment to support"),
("Patient","I keep thinking about my kids — what would they do if something happened to me.","fear",False,"fear of mortality and its impact on family"),
("Friend","We aren't going to go there yet. We take it one step at a time. Breathe.","serenity",False,"grounding and calming presence"),
]},
{"scenario":"workplace","topic":"anger_incompetence","utterances":[
("Senior","I've spent the last three days fixing the code that you said was 'finished' and 'tested'.","anger",False,"fury over colleague's laziness and lying"),
("Junior","It worked on my machine — I didn't realize the production environment was so different.","fear",False,"defensive and anxious excuse"),
("Senior","'It worked on my machine' is not a valid excuse for a senior-level project!","anger",False,"rejection of weak excuse"),
("Junior","I'm sorry — I'll stay late and help you finish the rest of the refactor.","submission",False,"attempted atonement"),
("Senior","You've already done enough damage — just go home and stay away from the repository until Monday.","anger",False,"dismissal due to loss of trust"),
("Junior","I understand. I'll be available via Slack if you need anything at all.","acceptance",False,None),
]},
{"scenario":"casual","topic":"surprise_revelation","utterances":[
("Daughter","I was looking through some old photos and I found out that my 'uncle' is actually my biological father.","surprise",False,"shock of major family secret revelation"),
("Friend","Wait — does your mother know that you've found this out yet.","apprehension",False,"concern for immediate family conflict"),
("Daughter","No, I'm still trying to process the fact that my entire life has been a lie.","surprise",False,"existential shock and disorientation"),
("Friend","That is a staggering amount of information to handle on your own.","trust",False,"offering a safe space to talk"),
("Daughter","I don't even know how to look at them tonight at dinner.","surprise",False,"fear of social interaction after shock"),
("Friend","You don't have to. Come over to my place and we'll figure out what to do next.","serenity",False,"providing an escape and calm"),
]},
{"scenario":"social","topic":"anticipation_wedding","utterances":[
("Bride","The doors are about to open and I can hear the music starting — this is actually happening.","anticipation",False,"excitement and nerves at start of wedding ceremony"),
("Maid_of_Honor","You look like an absolute queen and everyone out there is so excited to see you.","admiration",False,"genuine praise and support"),
("Bride","I'm so worried I'm going to trip on my dress or start sobbing in the middle of the vows.","apprehension",False,"fear of public embarrassment during ceremony"),
("Maid_of_Honor","If you trip, I'll be right there to catch you. Just look at him and forget everything else.","trust",False,"reassurance of support"),
("Bride","I can see him at the end of the aisle — he's smiling at me.","ecstasy",False,"intense joy and love"),
("Maid_of_Honor","Go get him! This is your moment!","joy",False,"celebration"),
]},
{"scenario":"casual","topic":"fear_intruder","utterances":[
("Resident","I just heard the back window break and someone is definitely walking around in the kitchen.","fear",False,"terror of home invasion in progress"),
("Dispatcher","I'm dispatching officers to your location right now. Can you lock yourself in a bedroom or a bathroom.","vigilance",False,"emergency safety instructions"),
("Resident","I'm in the closet and I can hear them opening the drawers right outside the door.","fear",False,"imminent threat and hiding in terror"),
("Dispatcher","Stay as quiet as possible. Do not come out until the officers identify themselves.","vigilance",False,"ensuring survival"),
("Resident","Oh god, I think they're trying the door handle — please help me.","fear",False,"extreme terror of being discovered"),
("Dispatcher","I have the officers on the scene — they are entering the building now. Stay down.","trust",False,"reassurance of rescue"),
]},
{"scenario":"workplace","topic":"anger_sabotage","utterances":[
("Developer_A","I just found a 'logic bomb' in the code that was clearly designed to trigger after I was fired.","anger",False,"fury over malicious technical sabotage"),
("Developer_B","It's a 'legacy feature' that was intended to 'ensure data consistency' in the event of a crash.","contempt",True,"sarcastic and deceptive technical excuse"),
("Developer_A","It was designed to delete the entire user table if your login was disabled!","anger",False,"specific proof of malicious intent"),
("Developer_B","You have a 'very active imagination' — maybe you should write fiction instead of code.","contempt",True,"sarcastic and dismissive reaction to being caught"),
("Developer_A","I've already sent the forensics report to the legal department and the police.","anger",False,"committing to a serious legal escalation"),
("Developer_B","I'm sure the 'police' will be 'fascinated' by your 'amateur debugging' — see you in court.","contempt",True,"sarcastic contempt for the legal system"),
]},
{"scenario":"casual","topic":"surprise_boost_1","utterances":[
("Friend_A","I just found out that my long-lost twin is actually a world-famous chef.","surprise",False,"shocking family discovery"),
("Friend_B","You've got to be kidding me — we've known each other for ten years!","amazement",False,"extreme amazement"),
("Friend_A","The DNA test came back this morning and it's a hundred percent match.","surprise",False,"confirming the shock"),
("Friend_B","I'm speechless — this is like something out of a daytime drama.","surprise",False,"speechless from amazement"),
("Friend_A","He's flying into the city tomorrow to meet me for the first time.","anticipation",False,"expecting a major life event"),
("Friend_B","I'm coming with you — I need to see this with my own eyes.","vigilance",False,"witnessing the event"),
]},
{"scenario":"casual","topic":"anticipation_boost_1","utterances":[
("Artist","The gallery is opening in ten minutes and my heart is about to beat out of my chest.","anticipation",False,"nerves before an exhibition"),
("Manager","The line of people waiting to get in is wrapped around the entire block.","joy",False,"success and excitement"),
("Artist","I've been working on these paintings for two years and now the world finally gets to see them.","anticipation",False,"high stakes expectation"),
("Manager","They're going to love them — you're the most talked-about artist in the city right now.","trust",False,"professional support"),
("Artist","They're opening the doors! Here we go!","anticipation",False,"climax of anticipation"),
("Manager","Take a deep breath and go greet your public.","serenity",False,"calming the artist"),
]},
{"scenario":"casual","topic":"ecstasy_boost_1","utterances":[
("Athlete","I just crossed the finish line and I've broken the world record by three full seconds.","ecstasy",False,"intense joy at historic victory"),
("Coach","I knew you could do it! All those four AM runs finally paid off!","ecstasy",False,"shared triumph"),
("Athlete","I've never felt so light and so completely alive in my entire life.","ecstasy",False,"visceral peak experience"),
("Coach","Look at the scoreboard — the whole stadium is on its feet for you.","ecstasy",False,"witnessing glory"),
("Athlete","I'm actually crying — I can't believe this is real.","ecstasy",False,"emotional peak"),
("Coach","It's real, and you're the champion of the world!","ecstasy",False,"final affirmation"),
]},
{"scenario":"casual","topic":"surprise_boost_2","utterances":[
("Traveler","I just walked into the hotel and they've upgraded us to the presidential suite for free.","surprise",False,"shock of unexpected luxury"),
("Partner","Wait — did you say the presidential suite? The one with the private rooftop pool?","surprise",False,"disbelief at the news"),
("Traveler","Yes! They said it's because it's our anniversary and they had a last-minute cancellation.","joy",False,"happy explanation"),
("Partner","I'm stunned — I've never even seen a room that expensive in person before.","surprise",False,"amazement at the scale of surprise"),
("Traveler","I'm already ordering the most expensive thing on the room service menu!","ecstasy",False,"peak indulgence"),
("Partner","Let's just spend the whole weekend in the pool and never leave!","joy",False,"celebration"),
]},
{"scenario":"casual","topic":"anticipation_boost_2","utterances":[
("Scientist","The telescope is finally aligned and we're about to receive the first images from the deep-field scan.","anticipation",False,"expecting a scientific breakthrough"),
("Colleague","We've spent a decade building this instrument — this is the moment of truth.","vigilance",False,"solemn focus on the result"),
("Scientist","The data stream is starting! The image is rendering on the screen right now.","anticipation",False,"heightened expectation during rendering"),
("Colleague","Look at those galaxies — they're clearer than anything we've ever seen before.","amazement",False,"wonder at the result"),
("Scientist","We're looking at the beginning of the universe!","ecstasy",False,"intense joy at discovery"),
("Colleague","This is going to change everything we know about physics.","anticipation",False,"expecting a major shift in knowledge"),
]},
{"scenario":"casual","topic":"ecstasy_boost_2","utterances":[
("Musician","The crowd is singing my lyrics back to me so loudly that I can't even hear the band.","ecstasy",False,"intense joy of musical connection"),
("Drummer","We've finally made it! Ten years in a van and now we're playing to eighty thousand people!","ecstasy",False,"shared peak experience"),
("Musician","I feel like I'm floating above the stage — the energy in this place is electric.","ecstasy",False,"visceral sense of peak performance"),
("Drummer","This is what we dreamed about when we were practicing in your mom's garage.","joy",False,"nostalgic happiness"),
("Musician","I never want this song to end!","ecstasy",False,"desire to stay in the peak"),
("Drummer","Let's give them an encore they'll never forget!","ecstasy",False,"furthering the peak"),
]},
{"scenario":"casual","topic":"surprise_boost_3","utterances":[
("Homeowner","I was stripping the wallpaper in the bedroom and I found a hidden safe behind the plaster.","surprise",False,"shock of hidden discovery"),
("Spouse","You're joking — is it empty or is there actually something inside it?","surprise",False,"disbelief and curiosity"),
("Homeowner","It's filled with gold coins and jewelry from the nineteen-twenties.","surprise",False,"shock of finding treasure"),
("Spouse","I'm actually light-headed — we just bought this house for a hundred thousand dollars!","surprise",False,"amazement at the value"),
("Homeowner","I think we just became very, very wealthy in the last ten minutes.","ecstasy",False,"peak financial joy"),
("Spouse","We have to call a lawyer and an appraiser right now!","vigilance",False,"taking action after shock"),
]},
{"scenario":"casual","topic":"anticipation_boost_3","utterances":[
("Baker","The judges are tasting my cake right now and they've been whispering for five minutes.","anticipation",False,"nerves during a competition"),
("Assistant","The head judge just took a second bite — that's always a good sign.","optimism",False,"hopeful sign recognized"),
("Baker","If I win this, I can finally open my own bakery in the city center.","anticipation",False,"high-stakes future expectation"),
("Assistant","They're standing up — they're about to announce the winner.","anticipation",False,"climax of expectation"),
("Baker","I can't breathe — my heart is in my throat.","apprehension",False,"physical symptoms of anxiety"),
("Assistant","And the winner of the Grand Prix is... you!","ecstasy",False,"peak joy at victory"),
]},
{"scenario":"casual","topic":"surprise_boost_4","utterances":[
("Employee","I just checked my bank account and I received a fifty thousand dollar bonus that I wasn't expecting.","surprise",False,"shock of unexpected large bonus"),
("Colleague","Fifty thousand?! Are you sure it's not a decimal error in the accounting software?","amazement",False,"disbelief at the amount"),
("Employee","I called HR and they confirmed it's a special award for the patent we filed last month.","surprise",False,"confirming the positive shock"),
("Colleague","I'm absolutely floored — I didn't think this company even gave out awards like that.","surprise",False,"amazement at company policy"),
("Employee","I'm going to pay off my student loans in a single afternoon!","ecstasy",False,"relief and joy"),
("Colleague","We are going to the most expensive bar in the city tonight to celebrate!","joy",False,"celebration"),
]},
{"scenario":"casual","topic":"surprise_boost_5","utterances":[
("Grandchild","I was digging in the garden and I found a time capsule that my grandfather buried in 1950.","surprise",False,"discovery of historical artifact"),
("Parent","I didn't even know he lived in this house back then — let's see what's inside.","surprise",False,"disbelief and curiosity"),
("Grandchild","There's a pristine newspaper from the day I was born and a letter addressed to me.","surprise",False,"shock of personal connection across time"),
("Parent","I'm stunned — he must have planned this decades before you were even a thought.","amazement",False,"wonder at the planning"),
("Grandchild","I feel like I'm touching a part of history that was waiting just for me.","serenity",False,"peaceful connection to the past"),
("Parent","This is the most incredible thing we've ever found in this yard.","joy",False,"happiness at discovery"),
]},
]

def process_dialogues(dialogues):
    utterances = []
    dialogue_idx = 1
    
    for d in dialogues:
        scenario = d.get("scenario", "casual")
        topic = d.get("topic", "general")
        prev_emo = None
        
        for i, utt in enumerate(d.get("utterances", [])):
            # Unpack the 5-tuple
            speaker, text, emotion, sarcasm_flag, emotion_cause = utt
            
            # Emoji Injection: Append a relevant emoji to ~80% of utterances
            if random.random() < 0.8:
                possible_emojis = EMOJI_MAP.get(emotion, ["✨"])
                emoji = random.choice(possible_emojis)
                # Avoid double spacing if text already has trailing space
                text = text.strip() + " " + emoji
            
            # Lookup metadata
            ring_info = PLUTCHIK.get(emotion, {"ring": "mild"})
            plutchik_ring = ring_info["ring"]
            sentiment = POLARITY.get(emotion, "neutral")
            
            # Use IAA base score and generate a realistic confidence score
            iaa = IAA.get(emotion, 0.70)
            conf = round(max(0.5, min(1.0, iaa - random.uniform(0.0, 0.15))), 2)
            
            # Basic validation
            word_count = len(text.split())
            if word_count < 6:
                # If a handwritten dialogue has fewer than 6 words, we flag it but keep it.
                # In production you might want to pad or enforce this at the source.
                pass
                
            emotion_shift = False
            if i > 0 and emotion != prev_emo:
                emotion_shift = True
            prev_emo = emotion
            
            utterances.append({
                "dialogue_id": f"D{dialogue_idx:04d}",
                "turn_id": i + 1,
                "speaker": speaker,
                "text": text,
                "emotion": emotion,
                "emotion_ring": plutchik_ring,
                "scenario": scenario,
                "sarcasm_flag": sarcasm_flag,
                "emotion_cause": emotion_cause,
                "sentiment_polarity": sentiment,
                "utterance_word_count": word_count,
                "inter_annotator_agreement": iaa,
                "confidence_score": conf,
                "topic": topic,
                "emotion_shift": emotion_shift
            })
            
        dialogue_idx += 1
        
    return utterances

def stratified_split(utterances):
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(utterances)
    dialogue_ids = list(df['dialogue_id'].unique())
    random.shuffle(dialogue_ids)
    
    # Initial split: 80% train, 10% val, 10% test
    n_total = len(dialogue_ids)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    train_ids = set(dialogue_ids[:n_train])
    val_ids = set(dialogue_ids[n_train:n_train+n_val])
    test_ids = set(dialogue_ids[n_train+n_val:])
    
    def get_split(did):
        if did in train_ids: return "train"
        if did in val_ids: return "val"
        return "test"
        
    df['split'] = df['dialogue_id'].map(get_split)
    
    ALL_EMOTIONS = list(PLUTCHIK.keys())
    
    # Ensure minimum 5 examples per emotion in Val and Test
    for split_target in ["val", "test"]:
        target_ids = val_ids if split_target == "val" else test_ids
        for emotion in ALL_EMOTIONS:
            split_df = df[df['split'] == split_target]
            count = split_df[split_df['emotion'] == emotion].shape[0]
            
            while count < 5:
                # Find a training dialogue containing this emotion
                train_candidates = df[(df['split'] == 'train') & (df['emotion'] == emotion)]['dialogue_id'].unique()
                if len(train_candidates) == 0:
                    break # Cannot fulfill
                
                # Pick the first candidate
                candidate_id = train_candidates[0]
                
                # Move candidate from train to target split
                train_ids.remove(candidate_id)
                target_ids.add(candidate_id)
                df.loc[df['dialogue_id'] == candidate_id, 'split'] = split_target
                
                # Recalculate count
                split_df = df[df['split'] == split_target]
                count = split_df[split_df['emotion'] == emotion].shape[0]
                
    # Now assign back to utterances
    split_map = df.set_index(["dialogue_id", "turn_id"])["split"].to_dict()
    for u in utterances:
        u["split"] = split_map.get((u["dialogue_id"], u["turn_id"]), "train")
        
    return utterances

def verify_constraints(utterances):
    try:
        import pandas as pd
    except ImportError:
        print("Please run `pip install pandas openpyxl` first.")
        return
        
    df = pd.DataFrame(utterances)
    emo_counts = df["emotion"].value_counts()
    
    ALL_EMOTIONS = list(PLUTCHIK.keys())
    
    print("\n" + "="*30)
    print("VERIFICATION REPORT")
    print("="*30)
    
    failed = False
    
    for e in ALL_EMOTIONS:
        count = emo_counts.get(e, 0)
        if count < 30:
            print(f"⚠️ [Constraint Failure] Emotion '{e}' has {count} utterances (needs >= 30)")
            failed = True
        
    total_len = len(df)
    if total_len < 1500:
        print(f"⚠️ [Constraint Failure] Total utterances: {total_len} (needs >= 1500)")
        failed = True
    else:
        print(f"✅ Total utterances constraint met ({total_len}).")
    
    num_dialogues = df["dialogue_id"].nunique()
    if num_dialogues < 150:
        print(f"⚠️ [Constraint Failure] Total dialogues: {num_dialogues} (needs >= 150)")
        failed = True
    else:
        print(f"✅ Total dialogues constraint met ({num_dialogues}).")
    
    for split, req_count in [("train", 20), ("val", 5), ("test", 5)]:
        split_df = df[df["split"] == split]
        split_emo_counts = split_df["emotion"].value_counts()
        for e in ALL_EMOTIONS:
            count = split_emo_counts.get(e, 0)
            if count < req_count:
                print(f"⚠️ [Constraint Failure] '{e}' in {split} has {count} (needs >= {req_count})")
                failed = True
                
    max_count = emo_counts.max()
    min_count = emo_counts.min()
    if min_count > 0 and max_count / min_count > 10:
        print(f"⚠️ [Constraint Failure] Class Imbalance {min_count}:{max_count} is worse than 1:10 cap.")
        failed = True
        
    sarcasm_pct = df["sarcasm_flag"].mean()
    if sarcasm_pct < 0.15:
        print(f"⚠️ [Constraint Failure] Sarcasm is {sarcasm_pct*100:.1f}% (needs >= 15%)")
        failed = True
    else:
        print(f"✅ Sarcasm constraint met ({sarcasm_pct*100:.1f}%).")
        
    short_utterances = df[df["utterance_word_count"] < 6]
    if not short_utterances.empty:
        print(f"⚠️ [Constraint Failure] Found {len(short_utterances)} utterances with length < 6 words.")
        failed = True
    else:
        print("✅ Utterance minimum length constraint met.")

    print("="*30)
    if failed:
        print("Warning: Some P0/P1 dataset constraints were not met. Check the logs above. You may need to add more dialogues to the DIALOGUES array.")
    else:
        print("Success! All constraints verified successfully.")
    print("="*30 + "\n")

def apply_tokenization(utterances, model_name="bert-base-uncased"):
    """
    Converts raw text into token IDs and masks compatible with Transformer models.
    """
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for u in utterances:
        # Tokenize the text
        encoded = tokenizer(
            u["text"],
            padding="max_length",
            truncation=True,
            max_length=64, # Standard for short dialogue turns
            return_tensors=None # Keep as list for JSON/CSV compatibility
        )
        
        # Add these new technical columns to our dataset
        u["input_ids"] = encoded["input_ids"]
        u["attention_mask"] = encoded["attention_mask"]
        
    return utterances

def export_data(utterances, output_dir="data/processed/ERC"):
    import os
    try:
        import pandas as pd
    except ImportError:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(utterances)
    
    # 1. CSV
    csv_path = os.path.join(output_dir, "plutchik_v2_production.csv")
    df.to_csv(csv_path, index=False)
    
    # 2. JSONL
    jsonl_path = os.path.join(output_dir, "plutchik_v2_production.jsonl")
    with open(jsonl_path, "w") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record) + "\n")
            
    # 3. XLSX
    xlsx_path = os.path.join(output_dir, "plutchik_v2_production.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df[df['split'] == 'train'].to_excel(writer, sheet_name="Train", index=False)
        df[df['split'] == 'val'].to_excel(writer, sheet_name="Validation", index=False)
        df[df['split'] == 'test'].to_excel(writer, sheet_name="Test", index=False)
        
        # Build Summary Statistics
        stats_data = []
        ALL_EMOTIONS = list(PLUTCHIK.keys())
        for emo in ALL_EMOTIONS:
            emo_df = df[df["emotion"] == emo]
            stats_data.append({
                "Emotion": emo,
                "Ring": PLUTCHIK.get(emo, {}).get("ring", "unknown"),
                "Total Count": len(emo_df),
                "Train": len(emo_df[emo_df["split"] == "train"]),
                "Val": len(emo_df[emo_df["split"] == "val"]),
                "Test": len(emo_df[emo_df["split"] == "test"])
            })
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name="Statistics", index=False)
        
        # Build Metadata Legend
        legend_data = [
            {"Column": "dialogue_id", "Description": "Unique identifier for the dialogue"},
            {"Column": "turn_id", "Description": "Sequential turn number within the dialogue"},
            {"Column": "speaker", "Description": "Role-specific speaker name"},
            {"Column": "text", "Description": "The utterance text (min 6 words)"},
            {"Column": "emotion", "Description": "Plutchik emotion label"},
            {"Column": "plutchik_ring", "Description": "Plutchik ring (intense, primary, mild, dyadic)"},
            {"Column": "scenario", "Description": "Context scenario of the dialogue"},
            {"Column": "sarcasm_flag", "Description": "Boolean indicating if utterance contains sarcasm"},
            {"Column": "emotion_cause", "Description": "Trigger for the emotion, or null"},
            {"Column": "sentiment_polarity", "Description": "positive, negative, or neutral"},
            {"Column": "utterance_word_count", "Description": "Number of words in utterance"},
            {"Column": "inter_annotator_agreement", "Description": "Simulated annotator agreement (0-1)"},
            {"Column": "confidence_score", "Description": "Reliability of the label (0-1)"},
            {"Column": "topic", "Description": "One-word topic for the dialogue"},
            {"Column": "emotion_shift", "Description": "True if emotion differs from previous turn"},
            {"Column": "input_ids", "Description": "Token IDs compatible with Transformer models"},
            {"Column": "attention_mask", "Description": "Attention masks for the tokens"}
        ]
        legend_df = pd.DataFrame(legend_data)
        legend_df.to_excel(writer, sheet_name="Legend", index=False)
        
    print(f"Export complete:\n - {csv_path}\n - {jsonl_path}\n - {xlsx_path}")

if __name__ == "__main__":
    import json
    try:
        with open("scripts/nuanced_dialogues.json", "r") as f:
            generated_dialogues = json.load(f)
            # Transform dict back to tuple format
            for d in generated_dialogues:
                for i, u in enumerate(d["utterances"]):
                    d["utterances"][i] = tuple(u)
            DIALOGUES.extend(generated_dialogues)
    except Exception as e:
        print(f"Warning: Could not load generated dialogues: {e}")

    print("Processing handcrafted dialogues...")
    utterances = process_dialogues(DIALOGUES)
    
    print(f"Total utterances extracted: {len(utterances)}")
    
    # Bug 5 Fix: Deduplicate on text to prevent leakage
    df = pd.DataFrame(utterances)
    df = df.drop_duplicates(subset=["text"], keep="first")
    utterances = df.to_dict(orient="records")
    
    print("Step 2: Applying Tokenization (BERT-style)...")
    # This prepares the data for the actual model
    utterances = apply_tokenization(utterances)

    print("Step 3: Applying stratified split...")
    utterances = stratified_split(utterances)
    
    print("Verifying constraints against P0/P1 requirements...")
    verify_constraints(utterances)
    
    print("Exporting data to data/processed/ERC/...")
    export_data(utterances)
    print("Done. Ready for Hugging Face ingestion.")
