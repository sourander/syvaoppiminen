---
priority: 100
---

# Neuroverkot 101

TODO! Tähän tulee ainakin seuraavat asiat:

* Mikä on neuroverkko ja niiden lyhyt historia
* Miten neuroverkko toimii (perusasiat)
* Mihin neuroverkkoja käytetään
* Mitä eroa on training vs. inference

## Historia

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

Kaikki kuvan nuolet ovat painoja (weights). Lineaarialgebrassa näitä kutsuttaisiin kulmakertoimiksi (slope), mutta neuroverkoissa termi on paino. Koska meillä on 1 sisääntulo ja 3 neuronia, näiden välillä on `1 x 3` eli kolme painoa. Lisäksi kutakin vakiotermiä (bias) kohden on yksi paino, joten niitä on kolme lisää. Yhteensä painoja on siis kuusi. Toivon mukaan tämä alkaa kuulostaa tutulta, kun mietit Johdatus koneoppimiseen kurssin normaaliyhtälön matriisiesitystä, joka käsiteltiin [Hill Climbing](https://sourander.github.io/ml-perusteet/algoritmit/linear/hill_climbing/) osiossa. Käytännössä syntyvä kaava on siis:

$$
y = /text{todo}
$$

Käsittelemme aktivointifunktiot myöhemmin, mutta tässä välissä riittää hyväksyä, että kunkin neuronin laskema arvo syötetään tyypillisesti ReLu-aktivointifunktioon, joka palauttaa nollan, jos syöte on negatiivinen, ja syötteen itsensä, jos se on positiivinen.

![](../images/100_ShallowReLU.svg)

**Kuva 3s:** ReLu-aktivointifunktio. (CC-BY-NC-ND) [^udlbook]



## Mihin käytetään

Ehdotuksia lähteiksi:

- [Nvidia Copeland: What’s the Difference Between Deep Learning Training and Inference?](https://blogs.nvidia.com/blog/difference-deep-learning-training-inference-ai/)
- [Dilmegani: Top 50 Deep Learning Use Case & Case Studies](https://research.aimultiple.com/deep-learning-applications/)
- [Lex Fridman: Deep Learning Basics: Introduction and Overview](https://youtu.be/O5xeyoRL95U) (YouTube-video)

## Tehtävät

!!! question "Tehtävä: Tämä on testi"

    Tämä on tehtäväpaketin testi ja placeholder. Näitä tulee lisää myöhemmin.

## Lähteet

[^udlbook]: Prince, S. Understanding Deep Learning. The MIT Press. 2023. https://udlbook.github.io/udlbook/