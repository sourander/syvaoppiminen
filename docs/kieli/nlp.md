---
priority: 700
---

# Luonnollinen kieli

Tietojenkäsittelytieteissä luonnollisella kielellä tarkoitetaan kieliä kuten suomi, englanti, ranska tai japani, joita ihmiset käyttävät päivittäisessä viestinnässään. Tämä on vastakohta keinotekoisille kielille, kuten ohjelmointikielille (esim. Python, Java) tai merkintäkielille (esim. HTML, XML). Merkittävä ero näiden välillä on, että koneelle tarkoitetut kielet ovat tehty insinöörinäkökulmasta siten, että formaalit säännöt ovat muodostettu ensin. Konekieli on käyttökelpoinen vasta kun säännöt ovat tarkoin määritelty. Ihmisten käyttämät kielet ovat syntyneet päinvastoin: käyttö ensin, säännöt myöhemmin. [^dlwithpython]

> "As a result, while machine-readable language is highly structured and rigorous, natural language is messy—ambiguous, chaotic, sprawling, and constantly in flux."
>
> – François Chollet ja Matt Watson [^dlwithpython]

Termi *ambiguous* tarkoittaa, että luonnollisessa kielessä samalle asialle voi olla useita merkityksiä tai tulkintoja. Suomalaisittain kuuluisa esimerkki tästä on lyhyt lause: =="Kuusi palaa"== (*engl. the spruce is on fire / spruce returns / the number six is on fire / ... / six pieces*). Entäpä kuinka tulkitaan seuraava uutisotsikko:

> "Susi hyök­kä­si omis­ta­jan­sa kanssa pyö­rä­len­kil­lä olleen koiran kimp­puun kes­kel­lä asu­tus­ta Raa­hes­sa"
>
> – Pyhäjokiseutu 02.10.2024

![](../images/700_gemini-susi-quote.jpg)

**Kuva 1:** *Kirjaimellisesti tulkittu otsikko: "Susi hyökkäsi omistajansa kanssa pyörälenkillä olleen koiran kimppuun keskellä asutusta Raahessa.". Kuva luotu Gemini Nano Banana mallilla.*

Onko tilanne kenties ollut Kuvan 1 mukainen: susi ja hänen omistajansa olivat hyökkääjät, kun koiraraukka yritti pyöräillä karkuun? Vastaavia monitulkintaisia lauseita on Suomen Kuvalehden Jyvät & Akanat -palstalla viikoittain. Tässä kaksi tuoretta esimerkkiä lisäviihteenä 5/2026 numerosta: 

> "Nikotiinipussit muuttavat aivoja – kokenut lääkäri kertoo, miten pääset niistä pysyvästi eroon"
>
> – Iltasanomat 5.1.2026

> "Varkaudessa anastettiin omaisuutta marraskuussa"
>
> – Iltasanomat 7.1.2026

Täten lienee selvä, että koneellinen kielen käsittely haastavaa, mutta koska kieli on ihmisten pääasiallinen viestintäväline, on luonnollisen kielen käsittely (Natural Language Processing, NLP) keskeinen osa tietojenkäsittelyä ja tekoälyä. Käytännön sovelluksia ovat esimerkiksi **tekstin luokittelu** (spam, no spam), **konekäännökset** (ranska → suomi), **hakukoneet** ja **tekstin generointi** (*"Olipa kerran... ?"*). Näiden haastavien tehtävien suhteen suuret läpimurrot ovat varsin tuoreita, mutta yritystä on kuitenkin ollut viimeisen yli 60 vuoden ajan. [^dlwithpython]

## Historia

Nykyisten chatbottien suuret merkkipaalut ovat syntyneet lähitulevaisuudessa, mutta NLP:llä on pitkä historia. Historia voidaan jakaa kolmeen pääkauteen: säännöpohjainen (rule-based), tilastollinen (statistical) ja syväoppimiseen perustuva (deep learning-based) NLP – joka toki on myös tilastollista.

### Turingin koe

Kaiken luonnollisen kielen käsittelyn päämäärä ei ole välttämättä matkia ihmisen älykkyyttä, mutta tämä ajatus on kulkenut matkassa alusta asti – aivan kuten muussakin tekoälyn historiassa. Alan Turing julkaisi 1950 paperin otsikolla "Computing Machinery and Intelligence", jossa hän esitteli ajatuksen ==Turingin kokeesta== nimellä "The Imitation Game" [^turing1950]. Peli perustuu vanhaan seurapeliin, jolla salongissa on voinut viihdyttää vieraita: henkilöt A ja B istuvat erillään toisistaan, ja kolmas henkilö C esittää kysymyksiä kummallekin. A ja B ovat mies ja nainen, ja henkilön C tehtävä on päätellä kirjoitetun viestin perusteella, kumpi on kumpi. Turingin koe on sama asetelma, mutta A ja B ovat kone ja ihmminen. Jos C ei pysty luotettavasti erottamaan konetta ihmisestä, voidaan sanoa, että kone on läpäissyt Turingin kokeen. [^aimarketing]

Löydät internetistä helposti tätä koetta kritisoivaa sisältöä, ja myös väitteitä, että eri mallit ovat läpäisseet kokeen. Jos haluat tutustua aihepiiriin, kannattanee tutustua vuoden 2025 artikkeliin otsikolla *Large Language Models Pass the Turing Test*, jossa on vertailtu niin ELIZAa kuin tuoreita GPT-4.5-malleja. [^llmturing]

### ELIZA

Tunnetuin varhaisista *chatterbot*-sovelluksista on ELIZA, erityisesti skripti DOCTOR, joka kehitettiin 1960-luvulla MIT:ssä. Sen loi Joseph Weizenbaum, ja se simuloitsi Carl Rogerin asiakaskeskeistä psykoterapiaa. Ohjelma on nimetty fiktiivisen Eliza Doolittle hahmon mukaan, joka esiintyy George Bernard Shaw'n näytelmässä "Pygmalion" (ja myöhemmin musikaalissa "My Fair Lady"). ELIZA käytti yksinkertaisia sääntöjä ja avainsanojen tunnistusta vastatakseen käyttäjän syötteisiin, luoden vaikutelman ymmärryksestä. Kyseessä on siis *pattern-matching*-järjestelmä. [^demystifyingai]

> "There are even accounts of ELIZA’s responses being so human-like that it evoked emotional responses from people who forgot they were interacting with a computer, including Weizenbaum’s own secretary. This led to much discussion about ELIZA’s potential to pass the Turing Test, although there are no known accounts of ELIZA actually doing this."
>
> – Robert Barton ja Jerome Henry [^demystifyingai]

!!! tip "Kokeile!"

    ELIZA:sta löytyy JavaScript-toteutuksia, mutta yksi näppärä tapa saada pääsy siihen on APT-paketinhallinnasta löytyvä PERL-toteutus. Sen saat käyttöön näin:

    1. Luo Dockerfile. Katso sisältö alta.
    2. Aja `docker build -t eliza .`
    3. Aja `docker run -it --rm eliza`

    ??? note "Dockerfile"

        ```Dockerfile
        FROM ubuntu:latest

        ENV DEBIAN_FRONTEND=noninteractive

        RUN apt-get update && \
            apt-get install -y --no-install-recommends \
                perl \
                libchatbot-eliza-perl && \
            rm -rf /var/lib/apt/lists/*

        WORKDIR /app

        RUN cat > eliza.pl << 'EOF'
        #!/usr/bin/perl
        use strict;
        use warnings;
        use Chatbot::Eliza;

        my $eliza = Chatbot::Eliza->new;

        print "Eliza: Hello. How can I help you today?\n";

        while (1) {
            print "You: ";
            my $input = <STDIN>;
            last unless defined $input;

            my $response = $eliza->transform($input);
            chomp $response;
            print "Eliza: $response\n";
        }
        EOF

        RUN chmod +x /app/eliza.pl

        ENTRYPOINT ["/app/eliza.pl"]
        ```

    ![ELIZA](../images/700_ELIZA-cli.png)

    **Kuva 1:** *ELIZA vastailee opettajan murheisiin.*

ELIZA herätti yleisön mielenkiinnon ja sen ympärille syntyi hypeä. Weizenbaum päätyi itse taistelemaan tätä hypeä vastaan, aloittaen ikään kuin ristiretken omaa ohjelmaansa vastaan. ELIZA ei tiennyt mitään psykologiasta ja oli pikemminkin Carl Rogerin terapian karikatyyri tai parodia, joka lähinnä heijastaa kysymykset takaisin käyttäjälle. [^aiux]

### PARRY

Toinen varhainen chatbot on PARRY, joka kehitettiin 1970-luvulla psykiatri Kenneth Colbyn Stanfordissa. Applikaatio pyrki simuloimaan skitsofreniaa sairastavan potilaan puhetta. PARRY käytti monimutkaisempia sääntöjä ja malleja kuin ELIZA, pyrkien jäljittelemään skitsofrenian oireita, kuten vainoharhaisuutta ja epäjohdonmukaista ajattelua. [^demystifyingai] Ohjelman LISP-koodi on linkitettynä Wikipedia-artikkelissa ja löytyy Carnegie Mellonin arkistosta. Sen ajaminen modernilla koneella olisi haastellista, joten tyydytään kurkkaamaan ELIZA:n (DOCTOR) ja PARRY:n keskustelua, joka on dokumentoitu vuonna 1972. Keskustelun alku muotoiltuna alla: [^rfc439]

:health_worker: **DOCTOR**: Good Evening. Tell me your problems.  Please terminate input with a period or a question mark.

:robot: **PARRY**: People get on my nerves sometimes.

:health_worker: **DOCTOR**: I am not sure I understand you fully.

:robot: **PARRY**: You should pay more attention.

:health_worker: **DOCTOR**: Suppose you should pay more attention.

:robot: **PARRY**: You're entitled to your own opinion.

:health_worker: **DOCTOR**: What makes you think I am entitled to my own opinion?

### Myöhemmät kehitysvaiheet

| Nimi                                       | Kehittäjä         | Vuosi | Huomioitavaa                   |
| ------------------------------------------ | ----------------- | ----- | ------------------------------ |
| ELIZA                                      | Joseph Weizenbaum | 1966  | Simuloi psykoterapeuttia       |
| PARRY                                      | Kenneth Colby     | 1972  | Simuloi skitsofreenikkoa       |
| [Jabberwacky](http://www.jabberwacky.com/) | Rollo Carpenter   | 1988  | Online-julkaisu 1997           |
| MS Word AutoCorrect                        | Microsoft         | 1993  | Yksinkertainen sääntöpohjainen |
| ALICE                                      | Richard Wallace   | 1995  | Käyttää AIML-skriptauskieltä   |
| MedSpeak                                   | IBM               | 1996  | Lääketieteellinen litterointi  |
| VAL                                        | BellSouth         | 1996  | Puheentunnistus ja vastaus     |
| SmarterChild                               | Robert Hoffer     | 2001  | Agentti AOL ja Messengerissä   |
| [Cleverbot](https://www.cleverbot.com/)    | Rollo Carpenter   | 2008  | Jabberwackyn seuraaja          |
| Siri                                       | Adam Cheyer       | 2011  | Assistentti                    |
| Xiaoice                                    | Microsoft         | 2014  | Empaattinen chatbot            |
| Alexa                                      | Amazon            | 2014  | Assistentti                    |
| Melody                                     | Andrew Ng         | 2015  | Lääketieteellinen assistentti  |


Taulukko on koostettu kirjoista *Conversational Artificial Intelligence* [^conversational] ja *The Invisible Brand: Marketing in the Age of Automation, Big Data, and Machine Learning*. [^aimarketing]


### Asiantuntijajärjestelmät (Expert Systems)

Heti alkuun suosittelen, että kannattaa *silmäillä* seuraavia videoita. Sinun ei välttämättä tarvitse katsoa pitkiä videoita kokonaan, mutta silmäilemällä näet, miten niissä käsitellyt asiantuntijajärjestelmät toimivat:

* [MIT: 3. Reasoning: Goal Trees and Rule-Based Expert Systems](https://youtu.be/leXa7EKUPFk). 50-minuuttinen ideo, jossa Patrick Winston esittelee Genesis-ryhmän tuottamaa Genesis-ohjelmaa, joka kykenee selostaa Macbeth-kirjan tapahtumia (aivan videon lopussa). Edeltävässä osio on pohjustavaa teoriaa.
* [URBS: Lecture 13: Building an Expert System and PyKE](https://youtu.be/mzsk5_EmZq8?si=SpVnrGcKvosEw58h). 50-minuuttinen luento, jossa esitellään PyKE-kirjasto ja rakennetaan asiantuntijajärjestelmä. Pitääkö ottaa sateenvarjo mukaan vai ei?

Asiantuntijajärjestelmät ovat säännöpohjaisia järjestelmiä, jotka käyttävät tietokantaa sääntöjä ja faktoja päätöksenteon tai ongelmanratkaisun tukena.

> "Expert systems are computer programs designed to mimic the decision-making abilities of human experts by leveraging predefined rules and knowledge bases."
>
> – Vijay Kanabar ja Jason Wong [^airevolution]

Jos asiantuntijajärjestelmällä haluaa analysoida tekstin tapahtumia, tulee käyttää jonkin sortin *semantic parser* -ohjelmaa, joka muuttaa luonnollisen kielen lauseet koneen ymmärtämään muotoon. Genesiksen kohdalla tämä on START-niminen ohjelma, joka kääntää englantia Genesiksen sisäiseen esitysmuotoon. Tämä selitetään *A Commonsense Approach to Story Understanding*-artikkelissa [^genesis]. [START](https://start.csail.mit.edu/index.php) itsessään on Boris Katz:n ja InfoLab:n (MIT) kehittämä kysymyksiin vastaava hakukone, mutta Genesis käyttää sitä vain parsijana. Sisäinen kieli sisältää entiteettejä (substantiiveja), suhteita (henkilö A verbi henkilö B), funktioita ja sekvenssejä. Genesis käyttää apunaan [ConceptNet](https://conceptnet.io/)-tietokantaa, *knowledge graph*:ia, jonka avulla voit esimerkiksi tutkia sanan [dog](https://conceptnet.io/c/en/dog) suhteita muihin käsitteisiin. Näiden päälle ovi käyttäjä rakaentaa sääntöjä, kuten `if XX harms YY, YY becomes angry` tai `if XX eats food, XX becomes full`. Jatkossa, lause `Matt eats an apple` istuu tähän sääntöön, koska ConceptNet yhdistää `apple`-sanan `food`-käsitteeseen (*is type of edible fruit*).

Toivon mukaan on tässä vaiheessa selvää, että olisi äärimmäisen haastavaa luoda modernin suuren kielimallin tasoinen asiantuntijajärjestelmä. Asiantuntijajärjestelmät olivat kovinta huutoa 80-luvulla. Modernien kielimallien kohdalla tulet kuitenkin törmäämään termeihin *knowledge graph* ja *ontology*. 

### Haasteita vs. MLP

90-luvulta alkoi selkeä siirtymä tilastollisiin menetelmiin ja perinteiseen koneoppimiseen. Tämän kurssin osalta tämä aihepiiri alkaa FC-MLP-verkoista (fully connected multilayer perceptron). Kuten on jo opittu, MLP-versiot eivät kykene tunnistamaan spatiaalisuutta. Tähän käytimme kuvien (ja äänen) kanssa konvoluutioverkkoja aiemmissa luvuissa. Lauseen voi kuvitella 1-D -spatiaaliseksi dataksi, jossa sanat ovat "pikseleitä" peräkkäin. Näin Conv1D-kerrokset soveltuvat tekstin käsittelyyn ainakin paperilla, ja niitä on siihen myös käytetty. Ongelmia, joita Conv1D-kerrokset eivät kuitenkaan ratkaise, ovat mm.:

1. **Sanat eivät ole lukuja**. Kuvissa pikselit ovat numeerisia arvoja (esim. 0–255), mutta sanat ovat kategorisia muuttujia. On ensin keksittävä tapa kääntää lauseet listaksi numeroita.
2. **Lauseiden pituudet vaihtelevat**. Yksi lause voi olla =="Kissa istuu matolla."== ja toinen =="Kissa matolla istui olevaista pohtien."== Konvoluutioverkot olettivat kiinteän syötteen pituuden.
3. **Pitkäaikaiset riippuvuudet**. Conv1D-kerrokset havaitsevat paikallisia kuvioita hyvin (esim. 2–5 peräkkäistä sanaa), mutta niiden on vaikea mallintaa pitkän kantaman riippuvuuksia. Kuvittele teos, joka alkaa sanoilla: *"Seuraavat 100 asiaa eivät ole totta: (1) ..."*.
4. **Konteksti ja monet merkitykset**. Sanat voivat saada merkityksensä kontekstin perusteella. Muista: "kuusi palaa".
5. **Taivutusmuodot**. Monet kielet sallivat sanojen taivuttamisen, eli ovat jossain määrin *morphologically rich*. Mitenpä suomen =="epäjärjestelmällistyttämättömyydellänsäkäänköhän"== ja =="epäjärjestelmällinen"== liittyvät toisiinsa?

Todella, todella naiivi ratkaisu yllä oleviin ongelmiin keittiöfilosofin pohdinnalla olisi:

1. ✅ Tee sanoista lukuja One-Hot Encoding -menetelmällä.
2. ✅ Täytä lauseet nollilla (padding) niin, että kaikilla on sama pituus.
3. ⛔ ???
4. ⛔ ???
5. ⛔ ???

Kolme viimeisintä jäisivät siis tyystin ratkaisematta – ainakin opettajan keittiöfilosofian taidoilla. Näihin ongelmiin onneksi löytyy parempia ratkaisuja, joihin pureudutaan tässä ja seuraavissa luvuissa.

## NLP:n perusteet

### Tekstin esikäsittely: SpaCy

Modernissa NLP:ssä esikäsittely on usein virtaviivaistettu valmiiden kirjastojen, kuten [SpaCy](https://spacy.io/):n, avulla. SpaCy ei ole ainut. Vaihtoehtoja olisivat esimerkiksi NLTK ja Gensim. SpaCy on kuitenkin suorituskykyinen ja helppokäyttöinen, joten keskitymme siihen.

Kun syötät tekstiä SpaCy-putkeen (*engl. pipeline*), se suorittaa taustalla automaattisesti useita komponentteja. Tämä ei ole kielen käsittelyn kurssi, joten keskitymme pääasiassa käyttämään valmiita putkia. Yksi valmiiksi koulutettu putki on [en_core_web_sm](https://spacy.io/models/en). Kyseessä ei ole yksittäinen tilastollinen malli, vaan joukko NLP-komponentteja, jotka on ketjutettu yhteen. Alla on taulukko, jossa nämä ovat selitettynä auki linkkeineen, sekä tieto siitä, onko kyseinen komponentti koulutetttava tilastollinen (lue: koneoppimiseen perustuva) malli vai sääntöpohjainen menetelmä (lue: pattern matching).

| Komponentti                                           | Tyyppi | Tunnistaa                          | Esim                                                 | Mistä löytyy tulos?              |
| ----------------------------------------------------- | ------ | ---------------------------------- | ---------------------------------------------------- | -------------------------------- |
| [Tokenizer](https://spacy.io/api/tokenizer)           | Rule   | N/A                                |                                                      | `Doc` itsessään                  |
| [tok2vec](https://spacy.io/api/tok2vec)               | ML     | N/A                                |                                                      | `Doc.tensor` tai `Doc[i].vector` |
| [tagger](https://spacy.io/api/tagger)                 | ML     | Sanaluokat                         | NN (noun; apple), PRP (pronoun; they), JJ (adj; big) | `Doc[i].tag_`                    |
| [parser](https://spacy.io/api/dependencyparser)       | ML     | Sanojen keskinäiset riippuvuudet   | nsubj (subject; she), prep (prep modifier; on)       | `Doc[i].dep_`                    |
| [ner](https://spacy.io/api/entityrecognizer)          | ML     | Erisnimet                          | ORG (organization; Google)                           | `Doc[i].ent_type_`               |
| [attributeruler](https://spacy.io/api/attributeruler) | Rule   | Poikkeussäännöt esim. sanaluokille |                                                      | N/A                              |
| [lemmatizer](https://spacy.io/api/lemmatizer)         | Rule   | Sanan perusmuoto                   |                                                      | `Doc.lemmas`                     |

!!! tip

    Kannattaa tutustua dokumentaatiosta sivuihin: 
    
    * [SpaCy Linguistic Features](https://spacy.io/usage/linguistic-features). Siellä käsitellään tarkemmin se, mikä on alla vain listattuna yhden esimerkin avulla.
    * [SpaCy Library Architecture](https://spacy.io/api). Tämä auttaa yllä olevan taulukon ymmärtämisessä visuaalisesti.
    * [SpaCy Training Pipelines & Models](https://spacy.io/usage/training). Tämä dokumentti selittää, miten SpaCy-mallit on koulutettu, ja paljastaa, mikä syväoppimiskirjasto sillä on käytössä konepellin alla.

    Jos haluaisit opetella SpaCyn syvällisemmin, kuten tehdä itse omia komponentteja putkeen, voisit aloittaa kurssin [Advanced NLP with spaCy](https://course.spacy.io/en/). Todennäköisesti haluaisit tutustua myös [YouTube: ExplosionAI-kanavaan](https://www.youtube.com/@ExplosionAI/), joka on SpaCyn kehittäjän kanava, sisältäen videoita sekä SpaCyn että Prodigyn käytöstä.


??? example "Kuinka ajaa alla olevat snippetit?"

    Sinulla pitää luonnollisesti olla SpaCy asennettuna (`uv add spacy`). Lisäksi sinun pitää ladata malli. Aja terminaalissa seuraava komento:

    ```bash
    uv add pip
    uv run spacy download "fi_core_news_sm"
    ```

    Tämän jälkeen käynnistä Marimo, luo uusi Notebook, ja ota malli käyttöön:

    ```python
    import spacy

    nlp = spacy.load("fi_core_news_sm")
    ```

#### Tokenisointi
Tekstin pilkkominen pienempiin yksiköihin, tokeneihin (sanat, välimerkit, erikoismerkit). Toisin kuin yksinkertainen `split(" ")`, älykäs tokenisoija ymmärtää esimerkiksi välimerkkien erottamisen sanoista.

```python
tokenized = nlp.make_doc("Kissa, se    on eläin?!")

for tok in tokenized:
    print(tok, end="|")
# Kissa|,|se|   |on|eläin|?|!|
```

#### Perusmuotoistaminen (Lemmatization)
Sanojen palauttaminen niiden sanakirjamuotoon eli perusmuotoon (esim. "juoksi" &rarr; "juosta", "kissojen" &rarr; "kissa"). Tämä on erityisen kriittistä suomen kielen kaltaisissa morfologisesti rikkaissa kielissä sanaston koon hallitsemiseksi.

```python
doc = nlp("Pienet pyöreät pippurit hyppivät")
for token in doc:
  print(token.lemma_, end=" ")
# pieni pyöreä pippuri hyppiä 
```

#### Sanaluokat (POS)
Jokaiselle tokenille ennustetaan sen sanaluokka (esim. substantiivi, verbi, adjektiivi). Onko kuusi numero vai mikä? Muuttuuko sanaluokka, jos annat lauseessa kontekstia, kuten kertomalla että *taskussani on kuusi pientä palaa kakkua*.

```python
doc = nlp("Kuusi palaa.")
for token in doc:
    print(f"{token.text} {token.pos_}")
# Kuusi NUM
# palaa VERB
```

#### Riippuvuussuhteet
Sanojen välisten syntaktisten suhteiden analysointi (engl. *syntactic dependency parsing*) – kuka tekee, mitä tekee, kenelle tekee. Tämä auttaa ymmärtämään lauseen rakennetta pintaa syvemmältä. Tähän löytyy jopa oma visualisointityökalu:

```python
from spacy import displacy

doc = nlp("Susi hyökkäsi omistajansa kanssa pyörälenkillä " 
+ "olleen koiran kimppuun keskellä asutusta Raahessa")

mo.Html(displacy.render(doc, style="dep"))
```

![](../images/700_displacy-susi-quote.png)

**Kuva 2:** *Riippuvuussuhteiden visualisointi SpaCy:llä.*

#### Nimettyjen entiteettien tunnistus (NER)
Errisnimien, organisaatioiden, paikkojen, päivämäärien ja rahasummien automaattinen tunnistus tekstivirrasta.

```python
doc = nlp("microsoft Microsoft MiCrOSofT MICROSOFT macrohard")

for token in doc:
    print(token.text, "==", token.ent_type_)
# microsoft == ORG
# Microsoft == ORG
# MiCrOSofT == 
# MICROSOFT == 
# macrohard == 
```

#### Morfologinen analyysi
Tunnistaa taivutusmuotoja.

```python
doc = nlp("Pöydällä")
doc[0].morph
# Case=Ade|Number=Sing
# eli adessiivi

doc = nlp("Pöydillä")
doc[0].morph
# Case=Ade|Number=Plur
# eli monikon adessiivi
```

#### Hukkasanat (Stop Words)
Hyvin yleisten ja usein merkitykseltään vähäisten sanojen (kuten "ja", "on", "että") suodattaminen pois, jotta malli voi keskittyä oleelliseen sisältöön.

```python
doc = nlp("Minua rassaa kun tuon niille jäätelöä, mutta kukaan ei niinku edes hymyile.")
for token in doc:
    print(token, end=" ") if not token.is_stop else print("---", end=" ")
# --- rassaa --- --- tuon --- jäätelöä , --- --- --- hymyile .
```

## Sanavektorit

!!! tip "Määritelmät"

    * **Embedding**: Yleinen termi, joka viittaa mihin tahansa tiheään numeeriseen esitykseen, joka säilyttää tietyn rakenteen tai suhteet alkuperäisessä datassa. Raschka:n mukaan se on *"a mapping from discrete objects, such as words, images, or even entire documents, to points in a continuous vector space"* [^llmfromscratch].
    * **Word Embedding**: Erityisesti sanojen tiheä numeerinen esitys, joka säilyttää semanttisia suhteita sanojen välillä. Esimerkiksi Word2Vec ja GloVe ovat menetelmiä, jotka luovat sanavektoreita. [^llmfromscratch]
    * **Word Vector** eli **sanavektori**: Sama kuin aiempi, mutta korostaa sen vektoriluonnetta matemaattisena objektina.

    Huomaa, että on siis muitakin embedding-tyyppejä, kuten lause- tai osasana- (engl. subword) embeddingit. Pahoittelut finglishistä: en löydä hyvää käännöstä embedding-sanalle.

SpaCy laskee sanavektoreita, joten kurkataan, kuinka niihin pääsee käsiksi. SpaCy:n suomenkielisen pipelinen tapauksessa vektori on 96-ulotteinen. Esimerkiksi sanaa `Pizza` kuvaa 96 featurea. Nämä featuret on opittu tilastollisesti valtavasta määrästä tekstiä.

```python
doc = nlp("Pizza on ravitsevaa.")
for token in doc:
    print(f"{token.text:>12}: ", end="")
    for value in token.vector[:3]:
        print(f"{value:>5.2f}", end=" | ")
    print(f"... | {value:>5.2f}")
```

Löydät vastaavan rakenteen myös `doc.tensor`-attribuutista, joka sisältää koko lauseen vektoritensorin, jonka muoto olisi tässä tapauksessa `(3, 96)`. PyTorchissa tulet käsittelemään niitä yleisimmin tensoreina kokoa: `(batch_size, seq_len, embedding_dim)`.


Mutta kuinka tähän outoon vektoriin ollaan päädytty? Tutustutaan alla eri menetelmiin, aloittaen Johdatus koneoppimiseen -kurssilta tutuksi tulleesta One-Hot Encoding -menetelmästä, edeten tiheisiin vektoreihin, joiden piirteet on opittu tilastollisesti. Seuraavassa luvussa tutustumme suurten kielimallien käyttämiin kontekstisidonnaisiin sanavektoreihin. Niiden ymmärtäminen on helpompaa, jos aloitetaan perusasioista.

### One-Hot Encoding

TODO! Esittele tässä baseline.

### Tiheät vektorit

TODO! Esittele termi Word2Vec (joka ei ole algoritmi vaan perhe). Alla on esiteltynä kaksi sen jäsentä: Word2Vec ja GloVe. Lisäksi esitellään fastText, joka on Facebookin paranneltu versio GloVe:stä.

#### GloVe

TODO! GloVe (Global Vectors) parantaa Word2Veciä hyödyntämällä koko korpuksen globaaleja yhteisesiintymistilastoja pelkän lokaalin ikkunan sijaan. Facebookin fastText vie tämän askeleen pidemmälle pilkkomalla sanat osiin (n-grams), jolloin malli pystyy ymmärtämään myös sille tuntemattomia sanoja niiden morfologisen rakenteen perusteella.

TODO! Korosta Word2Vec/GloVe-osiossa niiden heikkoutta: jos sanaa ei ole sanakirjassa (esim. kirjoitusvirhe tai harvinainen taivutusmuoto "syväoppimisellansakaan"), malli hajoaa tai käyttää <UNK>-tokenia. Tämä motivoi fastTextiä (joka on mainittu) ja myöhemmin BPE/WordPiece-tokenisaatiota Transfomer-luvussa.

#### fastText

TODO! Mainitse, että moderneissa NLP-putkissa on fastText:iä kehittyneempiä menetelmiä, kuten BERT-pohjaiset ratkaisut, tai jopa mallin sisäinen embedding-kerros, jotka oppivat kontekstisidonnaisia sanavektoreita mallin koulutuksen aikana. Näihin tutustutaan Transformers-luvussa.

TODO! Korosta staattisuutta. Sana "pankki" on aina sama vektori, oli kyseessä hiekkapankki tai Nordea.

### Vektorien vertailu

TODO! Kun sanat on muutettu numeerisiksi vektoreiksi, voimme laskea niiden välisiä etäisyyksiä selvittääksemme, mitkä sanat tai dokumentit ovat sisällöllisesti lähimpänä toisiaan.

TODO! Yleisin tapa mitata kahden sanavektorin samankaltaisuutta on laskea niiden välinen kulma (kosini), joka on riippumaton itse vektorin pituudesta (skaalasta). Käytännön harjoituksissa hyödynnämme tähän Pythonin SciPy-kirjaston spatial.distance.cosine -funktiota.

#### Vektorien analogiat

TODO! Vektoriavaruuden avulla voidaan laskea analogioita, kuten "kuningas - mies + nainen = kuningatar".

## Mallien arviointi (Metriikat)

TODO! Tekstiä tuottavien tai kääntävien mallien laadun mittaaminen on vaikeampaa kuin luokittelun, sillä "oikeita" vastauksia voi olla useita, ja siksi yksinkertainen tarkkuusprosentti (accuracy) ei riitä... tai ei ole edes määriteltävissä.

### Kielimallinnus: Perplexity

TODO! Perplexity (PPL) mittaa sitä, kuinka "hämmentynyt" tai epävarma kielimalli on ennustaessaan seuraavaa sanaa; matalampi arvo kertoo paremmasta kyvystä mallintaa kielen rakennetta. Matemaattisesti se voidaan johtaa mallin ristientropiasta ja on standardimittari perinteisille kielimalleille.

TODO! Perplexity voi olla vaikea käsite; sitä kannattaa ehkä avata intuitiolla: "Jos heität noppaa, perplexity on 6 (olet yhtä hämmentynyt kuin 1/6 todennäköisyys). Jos tiedät että noppa on painotettu antamaan aina kutosen, perplexity on 1."

### Konekäännös ja generointi: BLEU ja ROUGE

TODO! BLEU on konekäännösten standardimittari, joka laskee n-grammien päällekkäisyyttä koneen tuotoksen ja ihmisen tekemän referenssin välillä painottaen tarkkuutta (precision).

TODO! ROUGE on vastaava, erityisesti tiivistelmissä käytetty mittari, joka painottaa saantia (recall) eli sitä, kuinka suuri osa referenssitekstin sisällöstä löytyi koneen vastauksesta.


## Yhteenveto

TODO! Ennen siirtymistä RNN-lukuun, varmista että ero on selvä: NLP:ssä voidaan laskea sanavektorien keskiarvo (Bag-of-Words lähestymistapa, hukkaa sanajärjestyksen). Syväoppimisessa haluamme yleensä säilyttää järjestyksen (Sequence).

## Tehtävät

Tähän tulee tehtävät. Kirjoitan ne viimeiseksi.

## Lähteet

[^dlwithpython]: Watson, M & Chollet, F. *Deep Learning with Python, Third Edition*. Manning. 2025.
[^turing1950]: Turing, A. M. *Computing Machinery and Intelligence.* Mind. 1950. https://courses.cs.umbc.edu/471/papers/turing.pdf
[^aimarketing]: Ammerman, W. *The Invisible Brand: Marketing in the Age of Automation, Big Data, and Machine Learning*. McGraw-Hill. 2024.
[^llmturing]: Jones, C.R. & Benjamin, B. *Large Language Models Pass the Turing Test*. 2025. https://arxiv.org/abs/2503.23674
[^aiux]: Lew, G. & Schumacher, R. *AI and UX: Why Artificial Intelligence Needs User Experience*. Apress. 2020.
[^demystifyingai]: Barton, R. & Henry, J. *Demystifying Generative AI: A Practical and Intuitive Introduction*. Addison-Wesley Professional. 2026.
[^rfc439]: Unknown. *PARRY Encounters the DOCTOR*. 1973. https://www.rfc-editor.org/rfc/rfc439.html
[^conversational]: Rawat, R. et. al. *Conversational Artificial Intelligence*. Wiley-Scrivener. 2024.
[^airevolution]: Kanabar, V. & Wong, J. The AI Revolution in Project Management: Elevating Productivity with Generative AI*. Pearson. 2023.
[^genesis]: Williams, B. *A Commonsense Approach to Story Understanding*. MIT. 2016. https://groups.csail.mit.edu/genesis/papers/2017%20Bryan%20Williams.pdf
[^llmfromscratch]: Raschka, S. *Build a Large Language Model (From Scratch)*. Manning. 2024.