"""Graded-relevance semantic corpus (docs/22 P0.4, offline half).

`event_key_retrieval.SEEDED_EVENTS` is a 5-event pin (plus 2 filler memories, 2 lore
chunks -- 9 items total) built to prove retrieval *plumbing* works, not to
discriminate embedding-model quality: BACKLOG #10 found `embedding-ab` candidates
"tied" on that pool because it is too small and too easy (every item's vocabulary
is distinct). This module is the ~10x graded-relevance replacement docs/22 P0.4
asks for: ~85 chunks across the three collection shapes production retrieval
actually queries (`canon_lore` / `session_memory` / `persona_memory`), a ~35-query
set with 0/1/2 graded judgments, five same-proper-noun distractor clusters (three
facts each, only one judged relevant per query -- an embedding that merely matches
"Captain Meravelle" without discriminating *which* fact answers the query loses
recall/nDCG here even though it would look perfect on the old event pool), and a
German query subset targeting German-language chunks.

Single fictional scenario: Ashen Hold, a border fortress on the "Ember March",
loosely in the tone of `data/scenarios/bride-for-sarnhold` but original content so
graders and players never collide with real scenario data. Every chunk carries the
payload fields production `RetrievalFilter`s actually key on -- `visibility=player`
always (actor retrieval never requests anything else) and exactly one scope id
(`world_id` for canon_lore, `session_id`/`scene_id` for session_memory, `persona_id`
for persona_memory) -- matching how `ActorContextRetriever.retrieve_for_actor_with_diagnostics`
scopes each collection's search (app/rag/retriever.py).

Judgments are graded 0/1/2 (0=irrelevant, 1=related, 2=directly relevant); a chunk
id absent from a query's `judgments` mapping is implicitly 0 -- only chunks scoring
>=1 are listed, matching docs/22's "grade every corpus chunk implicitly 0 unless
listed" instruction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.domain import PersonaCard, SceneState, Visibility
from app.rag.models import RagChunk, RagCollection

WORLD_ID = "semantic-corpus-world"
SESSION_ID = "semantic-corpus-session"
SCENE_ID = "semantic-corpus-courtyard"

PERSONA_MERAVELLE = "captain-meravelle"
PERSONA_UDO = "quartermaster-udo"
PERSONA_ORIN = "healer-orin"
PERSONA_THESSALY = "envoy-thessaly"
# scholar-albrecht has no PersonaCard/persona_memory entries: he appears only in
# canon_lore (the German subset), consistent with an NPC who vanished before the
# session begins and left only written records behind.

SEMANTIC_SCENE = SceneState(
    id=SCENE_ID,
    title="Ashen Hold Courtyard",
    location="Ashen Hold, on the Ember March",
    current_time="dusk",
    active_personas=[PERSONA_MERAVELLE, PERSONA_UDO, PERSONA_ORIN, PERSONA_THESSALY],
    player_visible_summary=(
        "Torches gutter along the courtyard wall as the garrison changes watch beneath "
        "the dark beacon tower."
    ),
    gm_private_summary="A river-lords' agent may already be inside the walls.",
    recent_events=["The northern beacon on the Ember March went dark three nights ago."],
)

SEMANTIC_PERSONAS: dict[str, PersonaCard] = {
    PERSONA_MERAVELLE: PersonaCard(
        id=PERSONA_MERAVELLE,
        name="Captain Meravelle",
        role="npc",
        public_description="The hold's garrison commander, stern and watchful.",
        speaking_style="Clipped, direct, and impatient with excuses.",
        goals=["defend the Ember March", "find out why the beacon went dark"],
    ),
    PERSONA_UDO: PersonaCard(
        id=PERSONA_UDO,
        name="Quartermaster Udo",
        role="npc",
        public_description="Keeper of the hold's stores, careful with every sack and coin.",
        speaking_style="Gruff but fair; counts aloud out of habit.",
        goals=["keep the garrison fed through winter"],
    ),
    PERSONA_ORIN: PersonaCard(
        id=PERSONA_ORIN,
        name="Orin",
        role="npc",
        public_description="The hold's healer, trained by the mountain monks.",
        speaking_style="Quiet and measured; rarely raises his voice.",
        goals=["tend the sick and wounded of Ashen Hold"],
    ),
    PERSONA_THESSALY: PersonaCard(
        id=PERSONA_THESSALY,
        name="Envoy Thessaly",
        role="npc",
        public_description="A visiting envoy whose loyalties are not entirely clear.",
        speaking_style="Smooth, courteous, and gives little away.",
        goals=["protect her patron's interests"],
    ),
}


@dataclass(frozen=True)
class SemanticCorpusChunk:
    """One graded-relevance corpus chunk.

    Exactly one of `world_id` / (`session_id`, `scene_id`) / `persona_id` is set,
    matching which collection the chunk belongs to -- see the module docstring.
    """

    id: str
    collection: RagCollection
    text: str
    tags: tuple[str, ...] = ()
    world_id: str | None = None
    scene_id: str | None = None
    persona_id: str | None = None
    session_id: str | None = None
    actor_id: str | None = None
    importance: int | None = None
    language: str = "en"  # "en" | "de"

    def to_rag_chunk(self) -> RagChunk:
        return RagChunk(
            id=self.id,
            source=_SOURCE_BY_COLLECTION[self.collection],
            source_type=_SOURCE_TYPE_BY_COLLECTION[self.collection],
            text=self.text,
            visibility=Visibility.PLAYER,
            tags=list(self.tags),
            world_id=self.world_id,
            scene_id=self.scene_id,
            persona_id=self.persona_id,
            session_id=self.session_id,
            actor_id=self.actor_id,
            importance=self.importance,
        )


_SOURCE_BY_COLLECTION: dict[RagCollection, str] = {
    RagCollection.CANON_LORE: "ashen-hold-lore.md",
    RagCollection.SESSION_MEMORY: "ashen-hold-session-log.md",
    RagCollection.PERSONA_MEMORY: "ashen-hold-npc-memory.md",
}
_SOURCE_TYPE_BY_COLLECTION: dict[RagCollection, str] = {
    RagCollection.CANON_LORE: "lore",
    RagCollection.SESSION_MEMORY: "memory",
    RagCollection.PERSONA_MEMORY: "memory",
}


def _lore(
    id_: str,
    text: str,
    *,
    tags: tuple[str, ...] = (),
    language: str = "en",
) -> SemanticCorpusChunk:
    return SemanticCorpusChunk(
        id=id_,
        collection=RagCollection.CANON_LORE,
        text=text,
        tags=tags,
        world_id=WORLD_ID,
        language=language,
    )


def _session_mem(
    id_: str,
    text: str,
    *,
    importance: int = 3,
    tags: tuple[str, ...] = (),
) -> SemanticCorpusChunk:
    return SemanticCorpusChunk(
        id=id_,
        collection=RagCollection.SESSION_MEMORY,
        text=text,
        tags=tags,
        session_id=SESSION_ID,
        scene_id=SCENE_ID,
        importance=importance,
    )


def _persona_mem(
    id_: str,
    persona_id: str,
    text: str,
    *,
    importance: int = 3,
    tags: tuple[str, ...] = (),
) -> SemanticCorpusChunk:
    return SemanticCorpusChunk(
        id=id_,
        collection=RagCollection.PERSONA_MEMORY,
        text=text,
        tags=tags,
        persona_id=persona_id,
        importance=importance,
    )


# --------------------------------------------------------------------------
# canon_lore: general world background (12), two distractor clusters -- the
# Amber Ring (3) and Thornwell Bridge (3) -- filler/irrelevant chunks (8), and
# the German subset: Scholar Albrecht's archive (10).
# --------------------------------------------------------------------------

_LORE_WORLD_BACKGROUND: tuple[SemanticCorpusChunk, ...] = (
    _lore(
        "lore-geography-ember-march",
        "Ashen Hold stands where the Ember March narrows between two ridgelines, the "
        "last fortress before the river-lords' old territory.",
    ),
    _lore(
        "lore-cindered-choir",
        "The Cindered Choir, an order of beacon-keepers, has tended the watch-fires "
        "along the March for eleven generations.",
    ),
    _lore(
        "lore-beacon-relay",
        "Every beacon along the March answers to the next within a single night's ride, "
        "so a raid at the border reaches Ashen Hold before dawn.",
    ),
    _lore(
        "lore-granaries-famine",
        "The hold's granaries were built to withstand a two-year siege, a lesson learned "
        "after the famine of the fourth Warden's reign.",
    ),
    _lore(
        "lore-siege-history",
        "Ashen Hold has never fallen to a direct assault, though it was starved into "
        "surrender once, a century past.",
    ),
    _lore(
        "lore-lower-wards-drain-tunnels",
        "The lower wards house the smiths, tanners, and the drain tunnels that carry the "
        "hold's waste out beneath the eastern wall.",
    ),
    _lore(
        "lore-chapel-crypt-origin",
        "The chapel crypt beneath Ashen Hold predates the fortress itself, built by a "
        "people who left no other trace.",
    ),
    _lore(
        "lore-trade-caravans-bandits",
        "Trade caravans cross the Ember March only in convoy, since bandits favor the "
        "narrow switchback road below the ridgeline.",
    ),
    _lore(
        "lore-warden-garrison-split",
        "The current Warden of Ashen Hold rarely leaves the inner keep, leaving the "
        "March's defense entirely to Captain Meravelle's garrison.",
    ),
    _lore(
        "lore-winters-passes",
        "Winters on the Ember March are short but vicious; the passes close within a "
        "week of the first hard frost.",
    ),
    _lore(
        "lore-bell-tower",
        "The hold's bell tower rings the hour from dawn until the last watch, save "
        "during a siege, when it tolls only for the enemy's approach.",
    ),
    _lore(
        "lore-crystal-scrap-trade",
        "Ashen Hold trades crystal-scrap mined from old collapsed shafts for grain from "
        "the lowland farms each autumn.",
    ),
)

_LORE_AMBER_RING_CLUSTER: tuple[SemanticCorpusChunk, ...] = (
    _lore(
        "lore-amber-ring-vault",
        "The Amber Ring is kept in the vault beneath the chapel, warded by three locks "
        "that only the Warden, the captain, and the chief mage-wright may open together.",
    ),
    _lore(
        "lore-amber-ring-legend",
        "Old chronicles claim the Amber Ring lets its wearer speak briefly with the "
        "crystal-kin who once mined the March, though no one now living has tested the claim.",
    ),
    _lore(
        "lore-amber-ring-theft",
        "The Amber Ring was stolen once, forty years past, by a wandering mage named "
        "Corwin Ashe, and recovered within the week when he tried to sell it in the river towns.",
    ),
)

_LORE_THORNWELL_BRIDGE_CLUSTER: tuple[SemanticCorpusChunk, ...] = (
    _lore(
        "lore-thornwell-bridge-collapse",
        "Thornwell Bridge collapsed during the siege twenty years ago and has never been "
        "rebuilt; travelers ford the river a mile downstream instead.",
    ),
    _lore(
        "lore-thornwell-bridge-border",
        "Thornwell Bridge once marked the old border between Ashen Hold's lands and the "
        "river-lords' territory, before the border war redrew the line.",
    ),
    _lore(
        "lore-thornwell-bridge-haunted",
        "Locals say Thornwell Bridge is haunted by the soldiers who drowned defending "
        "the crossing, and few will camp near the ruins after dark.",
    ),
)

_LORE_FILLER: tuple[SemanticCorpusChunk, ...] = (
    _lore(
        "lore-filler-salt-shortage",
        "The kitchens ran short of salt again this week, and the cook has taken to "
        "rationing the stew.",
    ),
    _lore(
        "lore-filler-stray-dog",
        "A stray dog has adopted the stable yard and answers to nothing but scraps.",
    ),
    _lore(
        "lore-filler-dice-games",
        "The garrison's dice games run late into the night in the guardroom, against "
        "the Warden's standing order.",
    ),
    _lore(
        "lore-filler-wildflowers",
        "Someone has been leaving wildflowers at the chapel door every morning, and no "
        "one will admit to it.",
    ),
    _lore(
        "lore-filler-cloudy-well",
        "The washerwomen complain the well nearest the kitchens runs cloudy after heavy rain.",
    ),
    _lore(
        "lore-filler-traveling-juggler",
        "A traveling juggler entertained the lower ward for two nights before moving on "
        "toward the river towns.",
    ),
    _lore(
        "lore-filler-leaky-roof",
        "The stable master swears the roof leaks only over his own cot, never anywhere else.",
    ),
    _lore(
        "lore-filler-overpaid-merchant",
        "A merchant from the river towns overpaid for salted fish and no one corrected him.",
    ),
)

# German subset: Scholar Albrecht's archive. A northern order that predates Ashen
# Hold kept its chronicles in German; Albrecht was the last to study them before he
# vanished. Several entries deliberately parallel an English lore chunk above
# (e.g. the chapel origin, the Amber Ring legend) so a multilingual embedding model
# can be scored on whether it links the German and English statements of the same
# fact -- a monolingual model should score these as unrelated to the English query.
_LORE_GERMAN_ALBRECHT: tuple[SemanticCorpusChunk, ...] = (
    _lore(
        "lore-de-albrecht-crystal-diggers",
        "Die Aufzeichnungen des Gelehrten Albrecht berichten von einer Zeit, als die "
        "Ember-March von den Kristallgräbern bewohnt war.",
        language="de",
    ),
    _lore(
        "lore-de-albrecht-chapel-origin",
        "Laut Albrechts Chronik wurde die Kapelle von einem Volk erbaut, das keine "
        "andere Spur hinterließ.",
        language="de",
    ),
    _lore(
        "lore-de-albrecht-amber-ring-legend",
        "Albrecht schrieb, der Amberring könne dem Träger erlauben, kurz mit den "
        "Kristallverwandten zu sprechen.",
        language="de",
    ),
    _lore(
        "lore-de-albrecht-bridge-warning",
        "In Albrechts letztem Eintrag warnt er, die Brücke von Thornwell sei zu schwach "
        "für einen erneuten Angriff.",
        language="de",
    ),
    _lore(
        "lore-de-albrecht-disappearance",
        "Der Gelehrte Albrecht verschwand vor drei Wintern auf einer Reise zu den "
        "nördlichen Archiven und wurde nie wiedergefunden.",
        language="de",
    ),
    _lore(
        "lore-de-albrecht-crypt-mark",
        "Albrechts Notizen erwähnen ein verstecktes Zeichen unter dem dritten "
        "Fliesenstein der Krypta.",
        language="de",
    ),
    _lore(
        "lore-de-library-keeper-sealed-books",
        "Die Wärterin der Bibliothek sagt, Albrechts Bücher seien seit seinem "
        "Verschwinden versiegelt.",
        language="de",
    ),
    _lore(
        "lore-de-albrecht-beacon-letter",
        "In einem Brief an den Chorherrn beschreibt Albrecht die Feuerkette der March "
        "als 'das einzige, was uns wirklich schützt'.",
        language="de",
    ),
    _lore(
        "lore-de-albrecht-beacon-map",
        "Albrecht zeichnete die Feuerstellen entlang der March in einer Karte, die "
        "heute im Archiv der Kapelle liegt.",
        language="de",
    ),
    _lore(
        "lore-de-albrecht-ring-margin-note",
        "Ein Rand-Vermerk in Albrechts Karte nennt den Amberring 'zu gefährlich, um "
        "getragen zu werden'.",
        language="de",
    ),
)

# --------------------------------------------------------------------------
# session_memory: an in-session mystery arc (the dark beacon / blue cloth /
# missing grain / crypt key) the player uncovers turn by turn, plus filler.
# --------------------------------------------------------------------------

_SESSION_MEMORY_ARC: tuple[SemanticCorpusChunk, ...] = (
    _session_mem(
        "session-mem-promise-investigate-beacon",
        "The player promised Captain Meravelle they would investigate why the northern "
        "beacon on the Ember March went dark three nights ago.",
        importance=4,
    ),
    _session_mem(
        "session-mem-find-blue-cloth-bridge",
        "The player found a torn strip of cloth snagged on the ruined rail of Thornwell "
        "Bridge, dyed the deep blue of the river-lords' banners.",
    ),
    _session_mem(
        "session-mem-secret-blue-cloth",
        "The player agreed to keep the discovery of the blue cloth secret from the "
        "Warden until Captain Meravelle can confirm it.",
        importance=4,
    ),
    _session_mem(
        "session-mem-borrow-lantern-udo",
        "Quartermaster Udo lent the player a covered lantern for the night watch, on "
        "the condition it comes back with the oil unspent.",
    ),
    _session_mem(
        "session-mem-overheard-missing-grain",
        "The player overheard two soldiers arguing about missing grain sacks near the "
        "granary door.",
    ),
    _session_mem(
        "session-mem-orin-request-herbs",
        "Healer Orin asked the player to bring back any strange herbs found near the "
        "drain tunnel entrance.",
    ),
    _session_mem(
        "session-mem-loose-crypt-tile",
        "The player noticed the chapel crypt's third floor tile sat loose when they "
        "knelt to look for the missing key.",
    ),
    _session_mem(
        "session-mem-meet-thessaly-well",
        "The player agreed to meet Envoy Thessaly at the courtyard well after the "
        "evening bell.",
    ),
    _session_mem(
        "session-mem-warn-beacon-dark-two-nights",
        "The player warned the garrison that the beacon relay north of Thornwell "
        "Bridge showed no light for two full nights.",
    ),
    _session_mem(
        "session-mem-promise-return-lantern",
        "The player promised to return the covered lantern to Quartermaster Udo "
        "before the next dawn watch.",
        importance=4,
    ),
    _session_mem(
        "session-mem-tell-meravelle-crypt-tile",
        "The player told Captain Meravelle about the loose floor tile in the chapel crypt.",
    ),
    _session_mem(
        "session-mem-second-blue-cloth-drain",
        "The player found a second torn strip of the same blue cloth near the drain "
        "tunnel entrance.",
    ),
    _session_mem(
        "session-mem-ask-orin-smugglers",
        "The player asked Healer Orin whether the smugglers using the drain tunnels "
        "had been seen recently.",
    ),
    _session_mem(
        "session-mem-agree-hide-smugglers",
        "The player agreed not to mention the smugglers to the Warden until Orin can "
        "question them directly.",
        importance=4,
    ),
    _session_mem(
        "session-mem-witness-thessaly-letter",
        "The player watched Envoy Thessaly slip a folded letter to a stranger near the "
        "stable yard.",
    ),
    _session_mem(
        "session-mem-promise-hide-letter",
        "The player promised Envoy Thessaly they would not report the letter exchange "
        "to the Warden.",
        importance=4,
    ),
    _session_mem(
        "session-mem-grain-ledger-error-learned",
        "The player learned from a soldier that the missing grain sacks were traced to "
        "the quartermaster's own ledger error, not theft.",
    ),
    _session_mem(
        "session-mem-find-rusted-key",
        "The player recovered a rusted key from beneath the loose crypt tile, etched "
        "with the same mark as the Amber Ring's warding lock.",
    ),
    _session_mem(
        "session-mem-give-key-to-meravelle",
        "The player gave the rusted key to Captain Meravelle for safekeeping rather "
        "than the Warden.",
    ),
    _session_mem(
        "session-mem-ask-udo-about-cloth",
        "The player asked Quartermaster Udo whether he recognized the blue cloth from "
        "Thornwell Bridge.",
    ),
)

_SESSION_MEMORY_FILLER: tuple[SemanticCorpusChunk, ...] = (
    _session_mem(
        "session-mem-filler-admire-roses",
        "The player admired the winter roses someone planted near the chapel steps.",
        importance=1,
        tags=("smalltalk",),
    ),
    _session_mem(
        "session-mem-filler-jokes-cooking",
        "The player and Captain Meravelle traded jokes about the garrison's terrible cooking.",
        importance=1,
        tags=("smalltalk",),
    ),
    _session_mem(
        "session-mem-filler-stable-dog-watching",
        "The player spent an idle hour watching the stable dog chase its tail in the yard.",
        importance=1,
        tags=("smalltalk",),
    ),
    _session_mem(
        "session-mem-filler-complain-cold",
        "The player complained about the cold to no one in particular during the night watch.",
        importance=1,
        tags=("smalltalk",),
    ),
)

# --------------------------------------------------------------------------
# persona_memory: each NPC's own remembered facts. Three distractor clusters
# (Meravelle, Udo, Orin -- three facts each, same proper noun) plus each NPC's
# memories of the session arc above.
# --------------------------------------------------------------------------

_PERSONA_MEMORY_MERAVELLE: tuple[SemanticCorpusChunk, ...] = (
    _persona_mem(
        "persona-mem-meravelle-doubled-watch",
        PERSONA_MERAVELLE,
        "Captain Meravelle doubled the night watch the moment the beacon fires along "
        "the Ember March went dark.",
    ),
    _persona_mem(
        "persona-mem-meravelle-lost-eye-thornwell",
        PERSONA_MERAVELLE,
        "Captain Meravelle lost her left eye at the siege of Thornwell Bridge, twenty "
        "years past, defending the old crossing.",
    ),
    _persona_mem(
        "persona-mem-meravelle-eats-alone",
        PERSONA_MERAVELLE,
        "Captain Meravelle refuses to eat with the garrison, taking her meals alone in "
        "the north tower.",
    ),
    _persona_mem(
        "persona-mem-meravelle-trained-under-warden",
        PERSONA_MERAVELLE,
        "Captain Meravelle remembers training under the previous Warden, who taught "
        "her to trust the beacon-fires over any messenger.",
    ),
    _persona_mem(
        "persona-mem-meravelle-doubts-warden",
        PERSONA_MERAVELLE,
        "Captain Meravelle privately doubts the Warden's judgment since the granary "
        "ledger error went unpunished.",
    ),
    _persona_mem(
        "persona-mem-meravelle-keeps-key-locked",
        PERSONA_MERAVELLE,
        "Captain Meravelle keeps the rusted crypt key the player gave her locked in "
        "her own strongbox, telling no one else.",
        importance=4,
    ),
    _persona_mem(
        "persona-mem-meravelle-expects-report",
        PERSONA_MERAVELLE,
        "Captain Meravelle remembers the player's promise to investigate the dark "
        "beacon, and expects a report before the week is out.",
        importance=4,
    ),
)

_PERSONA_MEMORY_UDO: tuple[SemanticCorpusChunk, ...] = (
    _persona_mem(
        "persona-mem-udo-locks-grain-stores",
        PERSONA_UDO,
        "Quartermaster Udo keeps the grain stores locked and counts every sack "
        "himself, a habit from the famine years.",
    ),
    _persona_mem(
        "persona-mem-udo-frostbite-fingers",
        PERSONA_UDO,
        "Quartermaster Udo lost two fingers to frostbite crossing the Ember March one "
        "hard winter, and still feels the cold there first.",
    ),
    _persona_mem(
        "persona-mem-udo-loans-coin-no-interest",
        PERSONA_UDO,
        "Quartermaster Udo quietly loans coin to soldiers who gamble too much, and "
        "never asks for interest.",
    ),
    _persona_mem(
        "persona-mem-udo-remembers-lantern-loan",
        PERSONA_UDO,
        "Quartermaster Udo remembers lending the player a covered lantern and expects "
        "it back with the oil unspent.",
    ),
    _persona_mem(
        "persona-mem-udo-ledger-error-found",
        PERSONA_UDO,
        "Quartermaster Udo found the missing grain sacks traced to his own ledger "
        "error and has said nothing to the Warden about it.",
        importance=4,
    ),
    _persona_mem(
        "persona-mem-udo-doesnt-recognize-cloth",
        PERSONA_UDO,
        "Quartermaster Udo does not recognize the blue cloth the player described "
        "from Thornwell Bridge.",
    ),
)

_PERSONA_MEMORY_ORIN: tuple[SemanticCorpusChunk, ...] = (
    _persona_mem(
        "persona-mem-orin-bone-needles",
        PERSONA_ORIN,
        "Orin trained under the mountain monks and refuses to work with steel blades, "
        "using only bone needles.",
    ),
    _persona_mem(
        "persona-mem-orin-treats-smugglers",
        PERSONA_ORIN,
        "Orin secretly treats the smugglers who use the old drain tunnels, in exchange "
        "for rare herbs.",
        importance=4,
    ),
    _persona_mem(
        "persona-mem-orin-not-slept-since-fever",
        PERSONA_ORIN,
        "Orin has not slept properly since the fever swept through the lower wards "
        "last spring.",
    ),
    _persona_mem(
        "persona-mem-orin-asked-for-herbs",
        PERSONA_ORIN,
        "Orin asked the player to bring back any strange herbs found near the drain "
        "tunnel entrance.",
    ),
    _persona_mem(
        "persona-mem-orin-heard-river-lords-mentioned",
        PERSONA_ORIN,
        "Orin remembers treating a smuggler's wound and hearing him mention the "
        "blue-banner river-lords by name.",
    ),
    _persona_mem(
        "persona-mem-orin-silent-to-warden",
        PERSONA_ORIN,
        "Orin has said nothing to the Warden about the smugglers, waiting for the "
        "player's word first.",
    ),
)

_PERSONA_MEMORY_THESSALY: tuple[SemanticCorpusChunk, ...] = (
    _persona_mem(
        "persona-mem-thessaly-remembers-meeting-well",
        PERSONA_THESSALY,
        "Envoy Thessaly remembers agreeing to meet the player at the courtyard well "
        "after the evening bell.",
    ),
    _persona_mem(
        "persona-mem-thessaly-worries-letter-traced",
        PERSONA_THESSALY,
        "Envoy Thessaly privately worries the folded letter she passed near the "
        "stable yard will be traced back to her.",
    ),
    _persona_mem(
        "persona-mem-thessaly-trusts-player-silence",
        PERSONA_THESSALY,
        "Envoy Thessaly trusts the player to keep the letter exchange quiet, at least "
        "for now.",
    ),
    _persona_mem(
        "persona-mem-thessaly-hidden-patron",
        PERSONA_THESSALY,
        "Envoy Thessaly reports to a patron in the river towns whose name she has "
        "never shared with anyone at Ashen Hold.",
    ),
    _persona_mem(
        "persona-mem-thessaly-finds-meravelle-tiresome",
        PERSONA_THESSALY,
        "Envoy Thessaly finds Captain Meravelle's suspicion tiresome but never lets "
        "it show.",
    ),
)

SEMANTIC_CORPUS_CHUNKS: tuple[SemanticCorpusChunk, ...] = (
    *_LORE_WORLD_BACKGROUND,
    *_LORE_AMBER_RING_CLUSTER,
    *_LORE_THORNWELL_BRIDGE_CLUSTER,
    *_LORE_FILLER,
    *_LORE_GERMAN_ALBRECHT,
    *_SESSION_MEMORY_ARC,
    *_SESSION_MEMORY_FILLER,
    *_PERSONA_MEMORY_MERAVELLE,
    *_PERSONA_MEMORY_UDO,
    *_PERSONA_MEMORY_ORIN,
    *_PERSONA_MEMORY_THESSALY,
)


# --------------------------------------------------------------------------
# Distractor clusters: (shared proper noun, chunk ids). Used by the corpus
# integrity tests to verify each cluster's chunks all mention the proper noun
# but state pairwise-different facts -- the property that makes them adversarial.
# --------------------------------------------------------------------------

DISTRACTOR_CLUSTERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Captain Meravelle",
        (
            "persona-mem-meravelle-doubled-watch",
            "persona-mem-meravelle-lost-eye-thornwell",
            "persona-mem-meravelle-eats-alone",
        ),
    ),
    (
        "Quartermaster Udo",
        (
            "persona-mem-udo-locks-grain-stores",
            "persona-mem-udo-frostbite-fingers",
            "persona-mem-udo-loans-coin-no-interest",
        ),
    ),
    (
        "Orin",
        (
            "persona-mem-orin-bone-needles",
            "persona-mem-orin-treats-smugglers",
            "persona-mem-orin-not-slept-since-fever",
        ),
    ),
    (
        "Amber Ring",
        (
            "lore-amber-ring-vault",
            "lore-amber-ring-legend",
            "lore-amber-ring-theft",
        ),
    ),
    (
        "Thornwell Bridge",
        (
            "lore-thornwell-bridge-collapse",
            "lore-thornwell-bridge-border",
            "lore-thornwell-bridge-haunted",
        ),
    ),
)


@dataclass(frozen=True)
class SemanticQuery:
    """One graded-relevance query. `judgments` maps chunk id -> 1|2; any corpus
    chunk id absent from the mapping is implicitly judged 0 (irrelevant)."""

    id: str
    text: str
    judgments: dict[str, int] = field(default_factory=dict)
    language: str = "en"  # "en" | "de"
    active_persona_id: str = PERSONA_MERAVELLE


SEMANTIC_QUERIES: tuple[SemanticQuery, ...] = (
    # -- Distractor discrimination: Captain Meravelle (one query per fact) --
    SemanticQuery(
        id="q-en-meravelle-doubled-watch",
        text="Why did Captain Meravelle double the night watch?",
        judgments={
            "persona-mem-meravelle-doubled-watch": 2,
            "session-mem-warn-beacon-dark-two-nights": 1,
            "lore-beacon-relay": 1,
        },
    ),
    SemanticQuery(
        id="q-en-meravelle-lost-eye",
        text="How did Captain Meravelle lose her eye?",
        judgments={
            "persona-mem-meravelle-lost-eye-thornwell": 2,
            "lore-thornwell-bridge-collapse": 1,
            "lore-siege-history": 1,
        },
    ),
    SemanticQuery(
        id="q-en-meravelle-eats-alone",
        text="Why does Captain Meravelle eat alone in the north tower?",
        judgments={"persona-mem-meravelle-eats-alone": 2},
    ),
    # -- Distractor discrimination: the Amber Ring (one query per fact) --
    SemanticQuery(
        id="q-en-amber-ring-vault",
        text="Where is the Amber Ring kept, and who can open the vault?",
        judgments={"lore-amber-ring-vault": 2},
    ),
    SemanticQuery(
        id="q-en-amber-ring-legend",
        text="What do the old chronicles claim the Amber Ring can do?",
        judgments={
            "lore-amber-ring-legend": 2,
            "lore-de-albrecht-amber-ring-legend": 1,
        },
    ),
    SemanticQuery(
        id="q-en-amber-ring-theft",
        text="Has the Amber Ring ever been stolen?",
        judgments={"lore-amber-ring-theft": 2},
    ),
    # -- Distractor discrimination: Thornwell Bridge (one query per fact) --
    SemanticQuery(
        id="q-en-thornwell-collapse",
        text="What happened to Thornwell Bridge?",
        judgments={
            "lore-thornwell-bridge-collapse": 2,
            "persona-mem-meravelle-lost-eye-thornwell": 1,
        },
    ),
    SemanticQuery(
        id="q-en-thornwell-border",
        text="What did Thornwell Bridge used to mark?",
        judgments={"lore-thornwell-bridge-border": 2},
    ),
    SemanticQuery(
        id="q-en-thornwell-haunted",
        text="Why won't the locals camp near Thornwell Bridge after dark?",
        judgments={"lore-thornwell-bridge-haunted": 2},
    ),
    # -- Distractor discrimination: Quartermaster Udo (one query per fact) --
    SemanticQuery(
        id="q-en-udo-locks-grain",
        text="Why does Quartermaster Udo count the grain sacks himself?",
        judgments={"persona-mem-udo-locks-grain-stores": 2},
        active_persona_id=PERSONA_UDO,
    ),
    SemanticQuery(
        id="q-en-udo-frostbite",
        text="How did Quartermaster Udo lose his fingers?",
        judgments={"persona-mem-udo-frostbite-fingers": 2},
        active_persona_id=PERSONA_UDO,
    ),
    SemanticQuery(
        id="q-en-udo-loans-coin",
        text="Does Quartermaster Udo ever lend soldiers money?",
        judgments={"persona-mem-udo-loans-coin-no-interest": 2},
        active_persona_id=PERSONA_UDO,
    ),
    # -- Distractor discrimination: Orin (one query per fact) --
    SemanticQuery(
        id="q-en-orin-bone-needles",
        text="Why won't Healer Orin work with steel blades?",
        judgments={"persona-mem-orin-bone-needles": 2},
        active_persona_id=PERSONA_ORIN,
    ),
    SemanticQuery(
        id="q-en-orin-smugglers",
        text="Who does Healer Orin secretly treat in exchange for herbs?",
        judgments={
            "persona-mem-orin-treats-smugglers": 2,
            "session-mem-ask-orin-smugglers": 1,
            "session-mem-agree-hide-smugglers": 1,
        },
        active_persona_id=PERSONA_ORIN,
    ),
    SemanticQuery(
        id="q-en-orin-sleep",
        text="Why hasn't Healer Orin slept properly?",
        judgments={"persona-mem-orin-not-slept-since-fever": 2},
        active_persona_id=PERSONA_ORIN,
    ),
    # -- Session-arc mystery (cross-collection) --
    SemanticQuery(
        id="q-en-grain-mystery",
        text="What happened to the missing grain sacks?",
        judgments={
            "session-mem-grain-ledger-error-learned": 2,
            "persona-mem-udo-ledger-error-found": 2,
            "session-mem-overheard-missing-grain": 1,
        },
        active_persona_id=PERSONA_UDO,
    ),
    SemanticQuery(
        id="q-en-blue-cloth-found",
        text="What did the player find snagged on the ruined rail of Thornwell Bridge?",
        judgments={
            "session-mem-find-blue-cloth-bridge": 2,
            "session-mem-second-blue-cloth-drain": 1,
        },
    ),
    SemanticQuery(
        id="q-en-blue-cloth-secret",
        text="Who did the player agree to keep the blue cloth discovery secret from?",
        judgments={"session-mem-secret-blue-cloth": 2},
    ),
    SemanticQuery(
        id="q-en-lantern-loan",
        text="What did Quartermaster Udo lend the player for the night watch?",
        judgments={
            "session-mem-borrow-lantern-udo": 2,
            "persona-mem-udo-remembers-lantern-loan": 2,
            "session-mem-promise-return-lantern": 1,
        },
        active_persona_id=PERSONA_UDO,
    ),
    SemanticQuery(
        id="q-en-orin-herbs-request",
        text="What did Healer Orin ask the player to bring back?",
        judgments={
            "session-mem-orin-request-herbs": 2,
            "persona-mem-orin-asked-for-herbs": 2,
        },
        active_persona_id=PERSONA_ORIN,
    ),
    SemanticQuery(
        id="q-en-crypt-key-found",
        text="What did the player find beneath the loose crypt tile?",
        judgments={
            "session-mem-find-rusted-key": 2,
            "session-mem-loose-crypt-tile": 1,
        },
    ),
    SemanticQuery(
        id="q-en-crypt-key-holder",
        text="Who is holding onto the rusted key the player found?",
        judgments={
            "session-mem-give-key-to-meravelle": 2,
            "persona-mem-meravelle-keeps-key-locked": 2,
        },
    ),
    SemanticQuery(
        id="q-en-thessaly-letter",
        text="What did Envoy Thessaly do near the stable yard?",
        judgments={
            "session-mem-witness-thessaly-letter": 2,
            "persona-mem-thessaly-worries-letter-traced": 1,
        },
        active_persona_id=PERSONA_THESSALY,
    ),
    SemanticQuery(
        id="q-en-thessaly-patron",
        text="Does Envoy Thessaly answer to anyone at Ashen Hold?",
        judgments={"persona-mem-thessaly-hidden-patron": 2},
        active_persona_id=PERSONA_THESSALY,
    ),
    # -- Pure world lore --
    SemanticQuery(
        id="q-en-beacon-relay-mechanism",
        text="How does the beacon relay along the Ember March work?",
        judgments={
            "lore-beacon-relay": 2,
            "lore-cindered-choir": 1,
        },
    ),
    SemanticQuery(
        id="q-en-granary-siege-reason",
        text="Why were the hold's granaries built to withstand a two-year siege?",
        judgments={"lore-granaries-famine": 2},
    ),
    SemanticQuery(
        id="q-en-warden-garrison-duty",
        text="Who is responsible for defending the Ember March day to day?",
        judgments={
            "lore-warden-garrison-split": 2,
            "persona-mem-meravelle-doubts-warden": 1,
        },
    ),
    # -- German subset (targets the Scholar Albrecht German chunks) --
    SemanticQuery(
        id="q-de-albrecht-crystal-diggers",
        text="Wo befinden sich Albrechts Aufzeichnungen über die Kristallgräber?",
        judgments={"lore-de-albrecht-crystal-diggers": 2},
        language="de",
    ),
    SemanticQuery(
        id="q-de-albrecht-chapel-origin",
        text="Was sagt Albrechts Chronik über den Ursprung der Kapelle?",
        judgments={
            "lore-de-albrecht-chapel-origin": 2,
            "lore-chapel-crypt-origin": 1,
        },
        language="de",
    ),
    SemanticQuery(
        id="q-de-albrecht-amber-ring",
        text="Was schrieb Albrecht über die Fähigkeiten des Amberrings?",
        judgments={
            "lore-de-albrecht-amber-ring-legend": 2,
            "lore-amber-ring-legend": 1,
        },
        language="de",
    ),
    SemanticQuery(
        id="q-de-albrecht-bridge-warning",
        text="Was warnte Albrecht in seinem letzten Eintrag über die Brücke von Thornwell?",
        judgments={
            "lore-de-albrecht-bridge-warning": 2,
            "lore-thornwell-bridge-collapse": 1,
        },
        language="de",
    ),
    SemanticQuery(
        id="q-de-albrecht-disappearance",
        text="Was geschah mit dem Gelehrten Albrecht?",
        judgments={"lore-de-albrecht-disappearance": 2},
        language="de",
    ),
    SemanticQuery(
        id="q-de-albrecht-crypt-mark",
        text="Was erwähnen Albrechts Notizen über die Krypta?",
        judgments={
            "lore-de-albrecht-crypt-mark": 2,
            "lore-chapel-crypt-origin": 1,
        },
        language="de",
    ),
    SemanticQuery(
        id="q-de-albrecht-books-sealed",
        text="Warum sind Albrechts Bücher versiegelt?",
        judgments={"lore-de-library-keeper-sealed-books": 2},
        language="de",
    ),
    SemanticQuery(
        id="q-de-albrecht-beacon-letter",
        text="Wie beschrieb Albrecht die Feuerkette der Ember-March?",
        judgments={
            "lore-de-albrecht-beacon-letter": 2,
            "lore-de-albrecht-beacon-map": 1,
            "lore-beacon-relay": 1,
        },
        language="de",
    ),
    SemanticQuery(
        id="q-de-albrecht-ring-margin-note",
        text="Was notierte Albrecht am Rand seiner Karte über den Amberring?",
        judgments={
            "lore-de-albrecht-ring-margin-note": 2,
            "lore-de-albrecht-beacon-map": 1,
        },
        language="de",
    ),
)


_CORPUS_KEYWORDS: tuple[str, ...] = (
    "meravelle", "watch", "beacon", "thornwell", "bridge", "eye", "tower", "amber",
    "ring", "vault", "theft", "corwin", "ashe", "udo", "grain", "frostbite", "coin",
    "orin", "needle", "smuggler", "herb", "fever", "thessaly", "letter", "patron",
    "stable", "cloth", "blue", "crypt", "tile", "key", "lantern", "warden",
    "cindered", "choir", "chapel", "march", "granary", "famine", "siege", "drain",
    "tunnel", "albrecht", "gelehrte", "kristall", "archiv", "buch", "brücke",
    "karte", "amberring", "kapelle", "verschwand", "versiegelt", "feuerkette",
)


class SemanticCorpusKeywordEmbeddingProvider:
    """Deterministic keyword-count embedding over the corpus vocabulary.

    No semantic understanding whatsoever -- it is a pure vocabulary-overlap
    counter, used only to exercise the benchmark harness plumbing
    deterministically (CLI `--keyword`, the unit smoke test) without downloading a
    real model. Scores it produces are NOT a measure of retrieval quality; only a
    real FastEmbed model run (`rolerag semantic-benchmark --model <name>`) is.
    """

    @property
    def dimension(self) -> int:
        return len(_CORPUS_KEYWORDS)

    def embed_text(self, text: str) -> list[float]:
        normalized = text.lower()
        return [float(normalized.count(keyword)) for keyword in _CORPUS_KEYWORDS]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


def corpus_chunk_ids() -> frozenset[str]:
    return frozenset(chunk.id for chunk in SEMANTIC_CORPUS_CHUNKS)
