---
priority: 100
---

# Neuroverkot 101

## Historia

TODO

## Matalat neuroverkot

### Viittaus koneoppimiseen

Ennen kuin tutustumme aiheen syvääm päätyyn eli syviin neuroverkkoihin (engl. deep neural networks), on hyvä tarkistella metalia neuroverkkoja (engl. shallow neural networks). Kertaa alkuun Johdatus koneoppimiseen kurssilta [Normaaliyhtälö](https://sourander.github.io/ml-perusteet/algoritmit/linear/normal_equation/), [Gradient Descent](https://sourander.github.io/ml-perusteet/algoritmit/linear/gradient_descent/) sekä [Logistinen regressio](https://sourander.github.io/ml-perusteet/algoritmit/linear/logistic/). Kyseisellä kurssilla sinulle on kerrottu, että näissä aiheissa on pohja neuroverkkojen ymmärrykselle. Nyt on siihen paneutumisen aika.

Tavallisen 1D-regressiomallin rajoituksia ovat [^udlbook], että se voi mallintaa:

* vain viivan
* yhden inputin
* yhden outputin

Näitä rajoituksia kierrettiin Johdatus koneoppimiseen kurssilla osin käyttämällä logistista regressiota, SGD:tä ja polynomeja. Jos jälkimmäinen ei herätä muistikuvia, kertaa scikitin dokumentaatiosta [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html), jonka avulla muuttujista `[a, b]` voi muodostaa toisen asteen polynomifunktion `[1, a, b, a^2, ab, b^2]`. Malli on yhä lineaarinen parametrien suhteen, mutta muunnettu piirreavaruus mahdollistaa epälineaaristen kuvioiden mallintamisen alkuperäisessä syöteavaruudessa. Tämä ei ehkä ole tieteellisesti täysin pätevä vertaus, mutta voi auttaa: kuvittele, että piirrät ==logaritmiseen taulukkoon suoran viivan==. Viiva on lineaarinen logaritmisessa avaruudessa, mutta alkuperäisessä mittakaavassa ("todellisuudessa") se kuvaa eksponentiaalista käyrää.

### Määritelmä

Testaus kuvasta:

![](../images/100_ShallowTerminology.svg)

**Kuva 1:** Matala neuroverkko koostuu kerroksista: syöte (input), piilotettu (hidden) ja tuloste (output). Kerrokset yhdistävät eteenpäin suunnatut yhteydet (nuolet), joten näitä kutsutaan eteenpäin syöttäviksi verkoiksi (feed-forward networks). Kun jokainen muuttuja yhdistyy kaikkiin seuraavan kerroksen muuttujiin, kyseessä on täysin yhdistetty verkko. Yhteydet edustavat painokertoimia, piilokierroksen muuttujia kutsutaan neuroneiksi tai piiloyksikköiksi (hidden units). (CC-BY-NC-ND) [^udlbook]

Yllä näkyvän kuvan verkosta ==tekee matalan se, että siinä on vain yksi piilokerros==. Jos kerroksia olisi useita, kyseessä olisi syvä neuroverkko. Näihin tutustumme myöhemmin.

Nyt pyydän sinua palaamaan takaisin Johdatus Koneoppimiseen kurssin [Logistinen regressio](https://sourander.github.io/ml-perusteet/algoritmit/linear/logistic/) osioon. Siellä on esitelty logistinen regressio, joka on käytännössä yksi neuroni. Jos kytket syötteen useisiin neuroneihin, saat piilokerroksen. Jos kytket piilokerroksen useisiin neuroneihin, saat tulostekerroksen (output). Näin sinulla on neuroverkko luotuna. Muista, että tuloja ja lähtöjä voi olla useita.

### Mitä se tekee?

Käsitellään tämän otsikon alla seuraavanlaista verkkoa:

![](../images/100_ShallowNet.svg)

**Kuva 2:** Yksinkertainen neuroverkko, jossa on vain yksi syöte x, kolme piilotettua neuronia ja yksi tulos. Vasemmanpuoleiseen versioon on lisättynä vakiotermi (intercept, bias), joka yleensä jätetään kuvaajista pois. (CC-BY-NC-ND) [^udlbook]

Kaikki kuvan nuolet ovat painoja (weights). Lineaarialgebrassa näitä kutsuttaisiin kulmakertoimiksi (slope), mutta neuroverkoissa termi on paino. Koska meillä on 1 sisääntulo ja 3 neuronia, näiden välillä on `1 x 3` eli kolme painoa. Lisäksi kutakin vakiotermiä (bias) kohden on yksi paino, joten niitä on kolme lisää. Yhteensä painoja on siis kuusi. Toivon mukaan tämä alkaa kuulostaa tutulta, kun mietit Johdatus koneoppimiseen kurssin normaaliyhtälön matriisiesitystä, joka käsiteltiin [Hill Climbing](https://sourander.github.io/ml-perusteet/algoritmit/linear/hill_climbing/) osiossa. Kuvaa tutkimalla huomaat, että esimerkiksi $\theta_{10}$ ja $\theta_{11}$ vastaavat painoja, jotka yhdistävät syötteen $x$ ja vakiotermin $1$ piilotetun kerroksen ensimmäiseen neuroniin $h_1$. Theta on siis 3x2 matriisi, joka näyttää tältä:



$$
\Theta = \begin{bmatrix}
\theta_{10} & \theta_{11} \\
\theta_{20} & \theta_{21} \\
\theta_{30} & \theta_{31}
\end{bmatrix}
$$

Eli siis $h_1$, $h_2$ ja $h_3$, tai tarkemmin niiden esiasteet (pre-activation), lasketaan seuraavasti:

$$
\begin{align*}
h_{pre1} &= \theta_{10} + \theta_{11} x \\
h_{pre2} &= \theta_{20} + \theta_{21} x \\
h_{pre3} &= \theta_{30} + \theta_{31} x
\end{align*}
$$

Yllä olevissa lukee pienellä `pre`, koska kyseessä ovat esiasteet (pre-activations). Näistä saa varsinaiset piilotetun yksikön aktivoinnit (activations) aktivointifunktion avulla. Käsittelemme aktivointifunktiot myöhemmin kattavammin, mutta tässä välissä riittää hyväksyä, että kunkin piilotetun kerroksen neuronin laskema arvo syötetään tyypillisesti ReLu-aktivointifunktioon, joka palauttaa nollan, jos syöte on negatiivinen, ja syötteen itsensä, jos se on positiivinen.

![](../images/100_ShallowReLU.svg)

**Kuva 3s:** ReLu-aktivointifunktio. (CC-BY-NC-ND) [^udlbook]

Kun tämä aktivointifunktio, $f(z) = max(0, z)$, joka tunnetaan jatkossa a-merkkinä, on otettu huomioon, piilotetun kerroksen arvot ovat siis:

$$
\begin{align*}
h_1 &= a(\theta_{10} + \theta_{11} x) \\
h_2 &= a(\theta_{20} + \theta_{21} x) \\
h_3 &= a(\theta_{30} + \theta_{31} x)
\end{align*}
$$

Yllä olevassa kaavassa `x` on syöte, $\theta$ on painot ja $h$ on piilotetun kerroksen aktivoinnit eli varsinaiset *hidden unit* eli piiloyksiköt. Näiden lineaarinen yhdistelmä antaa tuloksen `y`:

$$
y = \phi_0 + \phi_1 h_1 + \phi_2 h_2 + \phi_3 h_3
$$

Kaiken kaikkiaan mallin opittua parametrejä ovat siis:

$$
\begin{align*}
\phi_0 &= \text{tuloskerroksen vakiotermi (bias)} \\
\phi_1, \phi_2, \phi_3 &= \text{tuloskerroksen painot} \\
\theta_{10}, \theta_{20}, \theta_{30} &= \text{piilotetun kerroksen muuttujien painot} \\
\theta_{11}, \theta_{21}, \theta_{31} &= \text{piilotetun kerroksen vakioiden painot}
\end{align*}
$$

Nämä neljä vaihetta, eli esiasteet, aktivoinnit, piilokerroksen lähtö ja viimeisen kerroksen tulos näkyvät alla olevassa kuvassa.

![](../images/100_ShallowBuildUp.svg)

**Kuva 4:** Neuroverkon laskennan vaiheet `a-j`. Viimeisen kuvaajan varjostetussa alueessa $h_2$ on passiivinen (leikattu), mutta $h_1$ ja $h_3$ ovat molemmat aktiivisia. (CC-BY-NC-ND) [^udlbook]


* **Esiasteet (a-c)**: Syöte x syötetään kolmeen lineaarifunktioon, joista jokaisella on eri y-leikkauspiste ja kulmakerroin.
* **Aktivoinnit (d-f)**: Jokainen lineaarifunktio syötetään ReLU-aktivointifunktioon, joka leikkaa negatiiviset arvot nollaan.
* **Painotus (g-i)**: Kolmea leikattu funktiota painotetaan (skaalataan) kertoimilla $\phi_1$, $\phi_2$ ja $\phi_3$.
* **Yhteenlasku (j)**: Leikatut ja painotetut funktiot summataan yhteen ja lisätään offset-arvo $\phi_0$, joka kontrolloi korkeutta.

Huomaa, että kuvaajassa on kolme "niveltä". Tästä tulee termi *piecewise linear function* (suom. paloittain määritelty lineaarinen funktio). Mikäli ennustettava ilmiö on monimutkainen, tarvitaan useampia piilokerroksia, jotta tämä paloittain määritelty funktio saadaan taiteltua haluttuun muotoon. Alla tästä vielä havainnollistava kuva.

![](../images/100_ShallowApproximate.svg)

**Kuva 5:** Katkoviivalla näkyvää todellista ilmiötä voi yrittää mallintaa eri piiloverkon kokoisilla malleilla. Vasemmanpuoleinen malli on selkeästi liian yksinkertainen, oikea on tarkka (joskin kenties liian tarkka.) (CC-BY-NC-ND) [^udlbook]

Jos haluat tutustua aiheeseen syvemmin, tutustu [Understanding Deep Learning](https://udlbook.github.io/udlbook/) kirjaan, joka on ilmainen ja avoin verkossa. Kirjaan liittyvä Qatarin yliopiston kurssi löytyy myös ilmaiseksi [YouTube: Deep Learning Fall 2024](https://youtube.com/playlist?list=PLRdABJkXXytCz19PsZ1PCQBKoZGV069k3&si=8FY_GMrQn0Pi8FPv).

## Koulutus ja inferenssi

Kurssilla käsitellään neuroverkkojen koulutusta ja inferenssiä (eli mallin käyttöä). Esitellään termit jo kuitenkin tässä alkuvaiheessa, koska ne ovat keskeisiä neuroverkkojen ymmärtämisessä, ja pohjustavat vastavirta (backpropagation) algoritmia, jota käsitellään parin seuraavan osion aikana.

Aiheesta löytyy myös pidempi Nvidian artikkeli, jos haluat tutustua: [What’s the Difference Between Deep Learning Training and Inference?](https://blogs.nvidia.com/blog/difference-deep-learning-training-inference-ai/)

### Koulutus

Koulutusvaiheessa mallille syötetään koulutusdataa. Tämä on Johdatus koneoppimisesta tuttu `X_train`-osuus datasetistä. Malli laskee ennusteet $\hat{y}$ ja vertaa niitä todellisiin arvoihin $y$. Näiden erotus lasketaan häviöfunktiolla (loss function). Häviöfunktio palauttaa yhden luvun, joka kertoo kuinka hyvin malli suoriutui. Tämän luvun perusteella mallin parametrejä säädetään, jotta häviö pienenee. Tätä toistetaan useita kertoja, kunnes malli on oppinut halutun tason tarkkuuden. Tämän pitäisi olla kertausta Johdatus koneoppimiseen -kurssilta. Kannattaa vilkaista omia muistiinpanoja ja omaa oppimispäiväkirjaa.

Neuroverkkojen koulutukseen liittyy yksi hyvinkin keskeinen ero perinteisiin malleihin verrattuna: Neuroverkot oppivat itse piirteet (feature learning). Perinteisissä malleissa piirteet piti usein valita käsin, mutta neuroverkot pystyvät oppimaan hyödylliset piirteet suoraan datasta.

Koulutuksen aikana tarvittu muistin määrä riippuu monesta tekijästä. Jos haluat tutustua asiaan klikkailemalla, käy kurkkaamassa interaktiivista Hugging Facen blogikirjoitusta [Visualize and understand GPU memory in PyTorch](https://huggingface.co/blog/train_memory).

### Inferenssi

Inferenssi on mallin käyttöä. Kun malli on koulutettu, se kirjoitetaan levylle: tai siis tarkemmin sanottuna sen parametrit tallennetaan. Jatkossa parametrit voidaan ladata käyttöön, jopa useille eri laitteille samanaikaisesti rinnakkain, ja mallia voidaan käyttää ennustamiseen. Tätä kutsutaan inferenssiksi.

Neuroverkkojen inferenssi vaatii vähemmän muistia (ja laskentatehoa) kuin koulutus, koska mallin parametrejä ei enää säädetä. Malli vain suorittaa eteenpäin syöttämisen (feed-forward) laskennan. Mallia voidaan myös eri tekniikoin pienentää ilman että suorituskyky kärsii liikaa. Näitä tekniikoita ovat esimerkiksi kvantisointi (quantization), pruneraus (pruning) ja *"tislaus tai tiivistys"* (distillation). Näitä käsitellään myöhemmin kurssilla ainakin pintapuolisesti. On hyvä kuitenkin jo tunnistaa, että mallin käyttökulut (inferenssi) ja koulutuskulut (training) eroavat toisistaan merkittävästi. Käytännössä voit törmätä vaikkapa [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)-julkaisun malliin BERT sivustolla Hugging Face siten, että osa vaatii enemmän ja osa vähemmän suorituskykyä. Alla taulukkona suuntaa-antava vertailu.

| Malli                                                                                                                                           | n parameteria | Tensor tyyppi | optimoinnin taso              | Muistin tarve painoille |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------- | ----------------------------- | ----------------------- |
| [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)                                                                       | 110M          | float32       | Alkuperäinen koulutettu malli | ~440 MB                 |
| [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                                                                       | 66M           | float32       | 40% pienempi, 60% nopeampi    | ~264 MB                 |
| [distilbert-base-uncased-distilled-squad](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad)                            | 66M           | float32       | QA-tehtäviin hienosäädetty    | ~264 MB                 |
| [distilbert-base-uncased-distilled-squad-int8-static-inc](https://huggingface.co/Intel/distilbert-base-uncased-distilled-squad-int8-static-inc) | 66M           | int8          | Kvantisointi                  | ~66 MB                  |

Muistin tarpeen voi laskea helposti: 32-bittinen liukuluku vaatii 4 tavua muistia. Näitä on 110 miljoonaa, joten 110M * 4B = 440MB. Kvantisoinnissa mallin painot muunnetaan 8-bittisiksi kokonaisluvuiksi, jolloin muistin tarve on vain neljäsosa alkuperäisestä. Huomaa, että inferenssissä muistia tarvitsee myös muita asioita, kuten syötteet, väliarvot ja mahdolliset välimuistit. Todellisen muistin tarve voi siis olla esimerkiksi 20-50 % enemmän kuin pelkkien painojen vaatima muisti.

## Mihin käytetään

Tutustu näihin:

- [Dilmegani: Top 50 Deep Learning Use Case & Case Studies](https://research.aimultiple.com/deep-learning-applications/)
- [Lex Fridman: Deep Learning Basics: Introduction and Overview](https://youtu.be/O5xeyoRL95U) (YouTube-video)

## Tehtävät

!!! question "Tehtävä: Tämä on testi"

    Tämä on tehtäväpaketin testi ja placeholder. Näitä tulee lisää myöhemmin.

## Lähteet

[^udlbook]: Prince, S. Understanding Deep Learning. The MIT Press. 2023. https://udlbook.github.io/udlbook/