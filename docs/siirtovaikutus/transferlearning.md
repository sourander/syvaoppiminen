---
priority: 610
---

# Siirtovaikutus

## Termi

### Pedagoginen tausta

Siirtovaikutus (engl. transfer learning) tarkoittaa aiemmin hankitun tieton tai taidon hyödyntämistä uudessa, usein suhteellisen samankaltaisessa tehtävässä. On hyvä huomata, että tämä termi ei ole alkujaan koneoppimisen kontekstissa syntynyt. Termi on kotoisin (kasvatus)psykologiasta. Pedagogiaa opiskelleet ovat lähes väistämättä törmänneet Thorndiken ja Woodworthin löydöksiin 1900-luvun alusta, jossa he osoittivat, että harjoittelun vaikutus yhdellä osa-alueella voi siirtyä toiseen osa-alueeseen, mikäli molemmissa on yhteisiä elementtejä. Kuten he kirjoittavat:

> "The general consideration of the cases of retention or of loss of practice effect seems to make it
> likely that spread of practice occurs only where identical elements are concerned in the
> influencing and influenced function." [^influence]

Oppiiko tammea osaava lapsi nopeammin shakin kuin ei-tammea osaava lapsi? Entä oppiiko shakkia taitava lapsia vertaisiaan nopeammin pelaamaan Carcassonnea? Nykyiset oppimiskäsitykset kasvatustieteissä ja kognitiotieteissä ovat epäilemättä kehittyneet Thorndiken ja Woodworthin ajoista, mutta koska tämän kurssin konteksti on koneoppiminen, emme syvenny pedagogisiin teorioihin enempää. Termin käyttö koneoppimisen yhteydessä on kuitenkin hyvin osuvaa, koska koneoppimisen siirtovaikutus perustuu pitkälti samankaltaiseen periaatteeseen. Siirtovaikutus ei ole *hopealuoti*, joka ratkaisee kaikki ongelmat aina, mutta samankaltaisen tehtävän äärellä se voi tarjota nopean ja tehokkaan tavan saavuttaa hyviä tuloksia. UKK-instituutin [Siirtovaikutus](https://tervekoululainen.fi/ylakoulu/liikuntataidot/siirtovaikutus/)-artikkeli on kirjoitettu liikunnan näkökulmasta, ja siinä kyseenalaistetaan jo otsikkotasolla: "Taidot siirtyvät vai siirtyvätkö?" [^tervekoululainen].

### Koneoppiminen

Koneoppimisen kontekstissa termi on syntynyt 1970-luvun aikoihin. Bozinovski koostaa historiallista taustaa artikkelissaan [^reminder], johon kannattaa tutustua ainakin pintapuoleisesti.

> "Basically it is using a pre-trained neural network
> (trained for Task1) for achieving shorter training time
> (positive transfer learning) in learning Task2." [^reminder]

On tärkeää erotta seuraavat termit:

* **Siirtovaikutus (Transfer Learning)**: Yleinen käsite eli kattotermi. Aiemmin opitun tiedon hyödyntämiseen uudessa tehtävässä [^aiengineering].
    * **Hienosäätö (Fine-tuning)**: Siirtovaikutuksen erityinen muoto, jossa esikoulutettua mallia säädetään uudelleen uudessa tehtävässä, usein pienemmällä oppimisnopeudella ja pienemmällä datamäärällä [^aiengineering].
    * **Piirrepohjainen siirto (Feature-based transfer)**: Siirtovaikutuksen muoto, jossa esikoulutetun mallin piirteitä käytetään suoraan uudessa tehtävässä ilman mallin painojen hienosäätöä [^aiengineering].

Tällä kurssilla käsittelemme erityisesti *piirrepohjaista siirtoa*. Toivon mukaan muistat konvoluutioverkoista sen, että ne oppivat matalammilla tasoilla yleisiä piirteitä, kuten reunoja ja muotoja, joista muodostetaan piirrevektori, jota käytetään luokitteluun. Tämä korvaa piirrevektorin käsin suunnitellut piirteet, joita käytettiin ennen syvien verkkojen aikakautta. Koska matalammat tasot oppivat yleisiä piirteitä, niitä voidaan hyödyntää uudessa tehtävässä, joka on samankaltainen kuin alkuperäinen tehtävä. Esimerkiksi esikoulutettu verkko, joka on koulutettu tunnistamaan esineitä ImageNet-datasetillä, voi olla hyvä lähtökohta hienosäätöön (fine-tuning) esimerkiksi lääketieteellisten kuvien luokittelussa. Tällöin verkon matalammat tasot säilytetään, ja vain mallin korkein taso, *fully connected layer*, eli varsinainen luokittelija, korvataan uudella.

Erityisesti kielimallien kontekstissa hienosäätö on yleinen käytäntö, jossa esikoulutettu malli, kuten GPT tai BERT, säädetään uudelleen tiettyyn tehtävään, kuten tekstin luokitteluun tai kysymys-vastaus -tehtäviin. Pohjalla on usein ohjaamatonta oppimista, tai tarkemmin *self-supervised learning* -lähestymistapa, jossa malli oppii kielen rakenteita ja merkityksiä suuresta tekstikorpuksesta ilman oikeaa vastausta (eli *label* puuttuu). Tämän jälkeen malli hienosäädetään pienemmällä, tehtäväkohtaisella datalla. Tähän tutustutaan kenties tällä kurssilla pintapuolisesti, mutta syvällisemmin myöhemmillä kursseilla.

Jos päädyt suorittamaan hienosäätöä, tarvitset enemmän muistia kuin pelkän loppuun liitetyn luokittelijan kanssa. Vastavirta-algoritmi tarvitsee tällöin kaikkien ei-lukittujen kerrosten painot, aktivoinnit ja gradientit. Tässä tulee äkkiä kotikäyttöisen GPU:n rajat vastaan. Tosin tyypillisesti aivan kaikkia kerroksia ei ylipäätänsä hienosäädetä: Watson ja Chollet suosittelevat ==partial fine-tuning== -lähestymistapaa, jossa vain verkon korkeimmat kerrokset säädetään uudelleen, ja matalammat kerrokset pidetään lukittuina. Tärkeimmiksi syiksi he nostavat, että (1) mallin alimmat kerrokset sisältävät geneerisiä muotoja ja piirteitä, jotka ovat kontekstista toiseen päteviä, ja (2) ylimääräisten painojen sisällyttäminen kouluttamiseen nostaa ylisovittamisen riskiä. [^dlwithpython]

## Käytännössä

Esikouluttamisen mallin käyttö piirteiden muodostamiseen on varsin helppoa. Tällöin käytännössä leikkaisit koko verkosta pois viimeisen luokittelukerroksen, ja käyttäisit jäljelle jäänyttä verkkoa piirteiden muodostamiseen. Otetaan esimerkiksi ResNet50, jossa 50 viittaa 50x painokerrosten määrään. Myös tämä malli on koulutettu aiemmin tutuksi tulleella ImageNet-datasetillä.

```python
conv_base = models.resnet50(weights='IMAGENET1K_V1')

# Tämä tulostaisi 49x painokerrosta - viimeisin on luokittelukerros nimeltään "fc.weight"
[name for name, child in conv_base.named_parameters() if "conv" in name]

# Meitä kiinnostaa enemmän verkon rakenne, joten kurkkastaan sen lapsia:
[x for x in conv_base.named_children()]
```

```plaintext title="Output"
[
  "conv1",
  "bn1",
  "relu",
  "maxpool",
  "layer1",
  "layer2",
  "layer3",
  "layer4",
  "avgpool",
  "fc"
]
```

Jos tutkimme pelkkää viimeistä kerrosta, sen rakenne on seuraava:

```python
conv_base.fc
```

```plaintext title="Output"
Linear(in_features=2048, out_features=1000, bias=True)
```

### Pelkät piirteet ulos

Jos haluaisimme tallentaa piirteet ulos, voisimme rakentaa uuden mallin, joka ottaa syötteenä kuvan ja palauttaa piirrevektorin:

```python
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        # Kopioi kaikki kerrokset paitsi viimeisen luokittelukerroksen
        self.features = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Litistä piirrevektori
        return x

feature_extractor = FeatureExtractor(conv_base)

# Usage
dummy_input = torch.randn(2, 3, 224, 224)
features = feature_extractor(dummy_input)
print(features.shape)  # Should be (2, 2048)
```

Nämä piirteet voi luonnollisesti tallentaa tiedostoon ja siten käyttää myöhemmin vaikkapa Logistic Regression -luokittelijan kouluttamiseen.

### Loppuun liitetty luokittelija

Jos haluaisimme rakentaa mallin, jossa on loppuun liitetty luokittelija, voisimme tehdä sen seuraavasti:

```python
new_head = nn.Linear(in_features=2048, out_features=num_classes)

conv_base.fc = new_head
```

!!! tip

    Huomaa, että voimme sisällyttää luokittelijaan myös piilotetun kerroksen, jolloin rakenne olisi hieman monimutkaisempi, kuten:

    ```python
    new_head = nn.Sequential(
        nn.Linear(in_features=2048, out_features=512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=512, out_features=num_classes)
    )

    conv_base.fc = new_head
    ```

Kun edellinen pää on korvattu uudella, meidän tulee kouluttaa vain tämä uusi pää. Tämä onnistuu asettamalla muiden kerrosten `requires_grad`-attribuutin arvoksi `False`:

```python
for param in conv_base.parameters():
    param.requires_grad = False

# Uuden fully connected -kerroksen gradientit pitää sen sijaan laskea
for param in conv_base.fc.parameters():
    param.requires_grad = True
```

Jatkossa malli koulutetaan aivan kuten kaikki aiemmatkin kurssin mallit: sinun tulee kirjoittaa *training loop*, jossa suoritetaan eteenpäin- ja taaksepäin-syöttö, sekä optimointivaihe.

### Partial Fine-Tuning

Jos haluaisimme suorittaa osittaista hienosäätöä, voisimme lukita vain osan verkon kerroksista. Esimerkiksi voisimme lukita kaikki kerrokset ennen `layer4`-kerrosta:

```python
for name, parameter in conv_base.named_parameters():
    parameter.requires_grad = False
    if 'layer3.5.bn3.bias' in name:
        break
```

On myös mahdollista tehdä tämä vaiheittain, eli kouluttaa $n$ epookin ajan vain luokittelijaa, ja sen jälkeen avata lisää kerroksia hienosäätöä varten.


### Toolkit

Ajoittain löydät myös valmiita työkaluja, jotka helpottavat siirtovaikutuksen hyödyntämistä. Esimerkiksi [hf:ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) on esikoulutettu Text-to-Speech-malli, jonka voi ladata ja hienosäätää helposti Hugging Facen `transformers`-kirjaston avulla. Online-yhteisö, tai tarkemmin kehittäjä Gokhan Eraslan, on julkaissut työkalun, jolla hienosäätö onnistuu helposti omalla datalla käyttöohjeineen: [gh:gokhaneraslan/chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning).


## Tehtävät

!!! question "Tehtävä: Dogs vs. Cats siirtovaikutus"

    Aja `610_dogs_vs_cats.py` Marimo Notebook.

    Kuinka korkealle sijoittautuisit alkuperäisessä [Kaggle: Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/leaderboard) -kilpailussa? Adrian Rosebrock esitteli muinoin kirjassaan, kuinka AlexNet:n kouluttamalla kokonaan ko. datasetillä saavutti noin 93 % tarkkuuden, mutta käyttämällä piirrepohjaista siirtovaikutusta (feature-based transfer learning) hän saavutti 98.69 % tarkkuuden, jolla olisi päässyt kilpailussa hopeasijalle! Mallina hänellä oli ResNet50, jolla hän generoi piirrevektorit, joita hän käytti Logistic Regression -luokittelijan syötteenä. [^dl4cv]

    Malli kouluttautui opettajan Macbookilla noin 8 minuuttia (5 epookkia).


!!! question "Tehtävä: RPS-datasetin luonti webcamilla"

    Laurence Monorey, joka on mm. kirjan *AI and ML for Coders in PyTorch* kirjoittaja, jakaa verkkosivuillaan datasettiä [Rock Paper Scissors](https://laurencemoroney.com/datasets.html), joka sisältää kuvia käsimerkeistä kivi, paperi ja sakset. Linkit ovat vanhoja, mutta alkuperäinen datasetti on yhä ladattavissa [storage.googleapis.com/learning-datasets/rps.zip](https://storage.googleapis.com/learning-datasets/rps.zip)-osoitteesta. Setissä on 840 kuvaa per luokka.
    
    Voisimme käyttää tätä CGI-generoitua datasettiä, mutta on paljon parmepaa oppimista kasata oma datasetti! Meillä kaikilla on jokin webcam, joten käyttäkäämme sitä. Tähän löytyy valmis toteutus skriptistä `611_rps_generator.py`, joka perustuu `wigglystuff`-kirjaston Marimo-widgettiin `WebcamCapture`. Skripti tallentaa kuvat `data/{label}/filename.jpg`-polkuun, jossa `{label}` on vakiona `rock`, `paper` tai `scissors` — on täysin sallittua käyttää muitakin luokkia, kuten `pehmolelu|kaukosäädin|kännykkä` tai `lasit|lippis|pipo`.

!!! question "Tehtävä: RPS ja Transfer Learning"

    Aja `612_rps_transfer_learning.py` Marimo Notebook. Skriptissä käytetään edellisessä tehtävässä tallennettua datasettiä ja koulutetaan GoogleNet Inception-v3 -mallia hyödyntäen loppuun liitettyä luokittelijaa. Malli on esikoulutettu ImageNet-datasetillä, joka sisältää 1000 luokkaa, mutta sinulla on vain 3 luokkaa (tai minkä verran niitä päätitkään tehdä).

    Tämän koulutus kestää vain joitakin sekunteja, olettaen että et kasaa valtavaa datasettiä.

## Lähteet

[^influence]: Thorndike, E. L. & Woodworth, R. S.. *The influence of improvement in one mental function upon the efficiency of other functions*. Psychological Review*,1901. https://doi.org/10.1037/h0074898
[^tervekoululainen]: UKK-instituutti. *Siirtovaikutus*. Terve koululainen. https://tervekoululainen.fi/ylakoulu/liikuntataidot/siirtovaikutus/
[^reminder]: Bozinovski, S. *Reminder of the First Paper on Transfer Learning in Neural Networks, 1976.* 2020. https://doi.org/10.31449/inf.v44i3.2828
[^aiengineering]: Huyen, C. *AI Engineering*. O'reilly. 2024.
[^dlwithpython]: Watson, M & Chollet, F. *Deep Learning with Python, Third Edition*. Manning. 2025.
[^dl4cv]: Rosebrock, A. *Deep Learning for Computer Vision with Python. Starter Bundle. 3rd Edition*. PyImageSearch. 2019.
