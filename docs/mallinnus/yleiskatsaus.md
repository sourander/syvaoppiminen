---
priority: 400
---

# Yleiskatsaus

## Mallinnus

Tämä luku on osin koostetta aiemmista luvuista, mutta aiheisiin sukelletaan hieman syvemmälle. Osa termeistä, kuten optimointifunktiot, ovat olleet vain sivulauseina. Leikkisästi sanottuna aihe on: *"Mistä on pienet – tai suuret – neuroverkot tehty?"* 🤖

Neuroverkkojen luomisen prosessista käytetään termiä *mallinnus* (*engl. modeling, building or developing a model*). Mallinnus tarkoittaa prosessia, jossa suunnitellaan, rakennetaan ja koulutetaan neuroverkko, joka ratkaisee tietyn ongelman. Mallinnus kattaa siis kaikki vaiheet datan hankinnasta ja esikäsittelystä mallin arkkitehtuurin suunnitteluun, kouluttamiseen ja arviointiin. Keskeisin vaihe on se, kun matemaattinen funktio sovitetaan dataan, mutta se on mekaanisin ja kenties helpoin osa-alue.

### Työvaiheet lyhyesti

Aiemmin Johdatus koneoppimiseen -kurssilla tutustuit jo yleiseen koneoppimisen työnkulkuun: datan keräämiseen, esikäsittelyyn, mallin valintaan ja arviointiin. Tässä luvussa tarkennamme näitä vaiheita nimenomaan neuroverkon näkökulmasta – eli siihen, millainen funktio valitaan ja miten sen muotoa (arkkitehtuuria) säädetään. Alla oleva taulukko on yhdistelmä Essential Math for AI -kirjan luvun 3 alun työnkulusta [^mathforai] sekä DEep Learning with Python (3rd ed) kirjan luvusta 6 [^dlwithpython]. Se kokoaa yhteen mallintamisen tärkeimmät vaiheet. Datan hankinta (tai lähinnä sen käyttö PyTorchissa) käsitellään tarkemmin seuraavassa luvussa. Datan louhintaa on käsitelty muilla kursseilla.


| Työvaihe                        | Kuvaus                                                                                                                              |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Tunnista **ongelma**            | Määrittele, onko kyse luokittelusta, regressiosta, generoinnista tai esimerkiksi anomalioiden havaitsemisesta.                      |
| Hanki sopiva **data**           | Varmista, että dataa on oikeanmallista, sitä on riittävästi ja se on laadukasta. Tämä vaihe on usein aikaa vievin.                  |
| Valitse **virhefunktio**        | Virhefunktio (eng. loss function, error function, cost function, objective function) kertoo, kuinka ohi tavoitteesta ennuste menee. |
| Luo **malli**                   | Suunnittele tai valitse hypoteesifunktio.                                                                                           |
| Valitse **optimointimenetelmä** | Gradienttimenetelmä (eng. gradient descent) on keskeinen työkalu minimointiin. Se perustuu virhefunktion derivaatan laskemiseen.    |
| **Regularisoi**                 | Regularisointi tekee funktiosta tasaisemman ja vähentää ylisovittamista.                                                            |
| **Kouluta**                     | Optimoi eli minimoi virhe . Tämä on CPU/GPU-intensiivinen *fit*-vaihe eli mallin koulutus.                                          |

## Työvaiheet yksityiskohtaisemmin

### Ongelma

> "You can’t do good work without a deep understanding of the context of what you’re doing. Why is your customer trying to solve this particular problem? What value will they derive from the solution? How will your model be used? How will it fit into your customer’s business processes? What kind of data is available or could be collected? What kind of machine learning task can be mapped to the business problem?" [^dlwithpython]
> 

Tämä on Johdatus koneoppimiseen -kurssilta tuttua. Esimerkiksi logistinen regressio soveltuu luokitteluun, kun taas lineaarinen regressio on tarkoitettu jatkuvien arvojen ennustamiseen, ja k-Means soveltuu klusterointiin. Neuroverkoissa tämä muuttuu sinänsä, että sama arkkitehtuuri voi soveltua monenlaisiin ongelmiin, kunhan mallin ulostulokerros ja virhefunktio valitaan oikein. Myöhemmin kurssilla käsittelemme kuviin, tekstiin tai aikasarjoihin erikoistuneita arkkitehtuureita; tämän viikon tarkoituksena on tunnistaa, kuinka ==eteenpäin kytketyn verkon (feedforward neural network)== arkkitehtuuri soveltuu erilaisiin ongelmiin.

Tulet huomaamaan, että sama Dense-kerrosten verkko soveltuu muiden muassa seuraaviin ongelmiin: regressio, binääriluokittelu, monen luokan luokittelu (joko yksi luokka tai useita luokkia kerrallaan) ja jopa epävarmuuden mallintaminen. Erona on vain ulostulokerros ja virhefunktio.

Ongelmaa pohtiessa on hyvä pyrkiä tunnistaa myös alin lähtökohta (engl. baseline). Luokittelussa alin lähtökohta on satunnainen arvaus: binääriluokittelussa 50 %, k-luokassa 1/k. Regressiossa alin lähtökohta tulee tunnistaa bisneskontekstissa: kuinka tarkka ennusteen tulee olla, jotta siitä on hyötyä?

### Data

Tähän palaamme seuraavassa luvussa. 

!!! tip

    Tässä vaiheessa on hyvä huomioida, että neuroverkon tulevat toimeen ns. raakadatan kanssa. Oletkin jo nähnyt, kuinka neuroverkko luokittelee kuvia pikseleiden intensiteettien perusteella ilman, että kuvia on erikseen muunnettu piirteiksi (kuten reunat, kulmat, muodot). Raakadata esitetään kuitenkin aina vektoroidussa numeromuodossa. Myös kielimallit toimivat numeroidulla datalla.

### Virhefunktio

Tässä on tärkeää erottaa kaksi asiaa: virhefunktio ja evaluointimittarit. Virhefunktion tulee soveltua optimointiin, eli sen tulee olla derivoituva. Evaluointimittarit taas voivat olla mitä tahansa, kunhan ne mittaavat haluttua ominaisuutta – mieluiten luotettavasti ja helposti tulkittavasti.

**Virhefunktio** (*engl. loss function, error function, cost function, objective function*) mittaa, kuinka hyvin malli suoriutuu annetusta tehtävästä.

**Evaluointimittarit** (*engl. evaluation metrics*) mittaavat mallin suorituskykyä, mutta niitä ei käytetä optimointiin. Esimerkiksi ROC AUC on suosittu binääriluokittelun arviointimittari, mutta sitä ei voi käyttää virhefunktiona, koska se ei ole derivoituva. [^dlwithpython]

Alla taulukko yleisimmistä tehtävätyypeistä, niiden ulostuloaktivoinneista ja virhefunktioista PyTorchissa. Taulukko on kirjoitettu englanniksi, koska en löydä vakiintuneita suomennoksia, jotka erottaisivat toisistaan moniluokkaisen ennustuksen kaksi eri tyyppiä (multi-class vs. multi-label classification). Pidemmiltä nimiltään nämä ovat ==multiclass, single-label classification== ja ==multilabel, multi-class classification==. Ensimmäisessä voi olla vain yksi luokka kerrallaan (esim. koira TAI kissa TAI lintu), kun taas jälkimmäisessä voi olla useita luokkia samanaikaisesti (esim. koira JA kissa).

| **Task type**                     | **Output activation**                | **PyTorch loss function** | **Task type explained**   | **Human metric**             |
| --------------------------------- | ------------------------------------ | ------------------------- | ------------------------- | ---------------------------- |
| **Regression**                    | None                                 | `MSELoss`, `L1Loss`       | `0.123`                   | MAE                          |
| **Binary classification**         | None (Sigmoid) :one:                 | `BCEWithLogitsLoss`       | `a` or not                | binary accuracy, F1, ROC AUC |
| **Multi-label classification**    | None (Sigmoid) :one:                 | `BCEWithLogitsLoss`       | `a` and/or `b` and/or `c` | binary accuracy, F1, ROC AUC |
| **Multi-class classification**    | None (Softmax) :two:                 | `CrossEntropyLoss`        | `a` or `b` or `c`         | top-k accuracy, ROC AUC      |
| **Gaussian regression** :three:   | None (mean), <br>Softplus (variance) | `GaussianNLLLoss`         | mean and variance         | MAE (mean only)              |
| **Poisson count modeling** :four: | None                                 | `PoissonNLLLoss`          | `0` or `1` or `2` … `n`   | MAE                          |

!!! note "Selitykset"

    :one: Saatat nähdä yhdistelmän `Sigmoid + BCELoss`, mutta `BCEWithLogitsLoss` on suositeltavampi, koska se on numeerisesti vakaampi (kuten [PyTorchin dokumentaatiossa: BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) todetaan.)

    :two: Saatat nähdä yhdistelmän `LogSoftmax + NLLLoss` moniluokkaisessa luokittelussa, mutta `CrossEntropyLoss` sisältää jo molemmat vaiheet ja välttää turhan työn.

    :three: Gaussinen regressiomalli antaa kaksi lukua: arvauksen (keskiarvo) ja epävarmuuden (varianssi). Malli voi esimerkiksi sanoa: *"Arvioin, että lämpötila on 21 °C, mutta epävarmuus on noin ±3 °C."* Varianssin positiivisuus voidaan varmistaa eksponentti- tai Softplus-aktivoinnilla. Softplus lisää numeerista vakautta, mutta tuo myös pienen harhan tai lattian varianssiin.

    :four: Poisson-malli olettaa, että ilmiö noudattaa Poisson-jakaumaa tavallisen normaalijakauman (kuten regressiossa) sijaan. Malli ennustaa ei-negatiivisen luvun: *"Annettujen piirteiden perusteella ennustan, että asiakkaita tulee 10."*


### Malli

Nyt sinulla pitäisi olla data vektoroituna, ongelma määriteltynä ja virhefunktio valittuna. Seuraava askel on hypoteesifunktion eli mallin valinta. Tätä varten sinun on käytännössä pakko pilkkoa datasetti kolmeen osaan:

* Koulutusdata (training set)
* Validointidata (validation set)
* Testidata (test set)

Hypoteesifunktiolla eli mallilla on monta nimeä: *"We use the terms hypothesis function, learning function, prediction function, training function, and model interchangeably."* [^mathforai] Johdatus koneoppimiseen -kurssilla mallin valinta oli yksi vaiheista; neuroverkkojen kohdalla mallin valinta tarkoittaa arkkitehtuurin määrittelyä. Käytännössä mallin kerrosten määrä ja muoto ovat hyperparametreja. "A deep learning model is like a sieve for data processing, made of a succession of increasingly refined data filters—the layers [^dlwithpython]".

Watson ja Chollet kirjoittavat, että "To figure out how big a model you’ll need, you must develop a model that overfits." [^dlwithpython] Tämä tarkoittaa, että et käytännössä voi etukäteen tietää, minkä kokoinen malli soveltuu tehtävään. Aloita pienestä ja skaalaa sitä suuremmaksi kunnes löydät ylisovittamisen rajan. Tämän jälkeen voit käyttää regularisointia ja etsiä tasapainoa alisovittamisen ja ylisovittamisen välillä. Malli monimutkaisuus määrittyy seuraavilla:

* Kerrosten lukumäärä
* Kunkin kerroksen solmujen (neuroneiden) lukumäärä
* Epookkien lukumäärä

Lisäksi voit muuttaa esimerkiksi aktivointifunktioita, optimointimenetelmiä ja oppimisnopeutta (learning rate). Näihin kuitenkin löytyy yleensä hyvät oletusarvot, joilla pääsee pitkälle. Tutustumme näihin tarkemmin [Kouluttamisen käytännöt](kaytannot.md)-luvussa.

### Regularisointi

Tähän palataan [Kouluttamisen käytännöt](kaytannot.md)-luvussa.

### Koulutus

Mallin varsinainen koulutus tapahtuu optimointimenetelmällä, joka minimoi virhefunktion arvon. Yleisin menetelmä on gradienttimenetelmä (engl. gradient descent). Gradienttimenetelmä perustuu virhefunktion derivaatan laskemiseen ja sen hyödyntämiseen mallin painojen päivittämisessä siten, että virhe pienenee jokaisella askeleella. Tämän pitäisi olla sinulle jo tuttua.

Muutoin tämä vaihe on jossain määrin mekaaninen. Käytät valmista kirjastoa, GPU alkaa tuottaa lämpöä ja sinä menet kahville. ☕

## Tehtävät

!!! question "Tehtävä: Mallinna Fashion MNIST"

    Kouluta yksinkertainen neuroverkko [FashionMNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html) -datasetillä. Datasetti on siitä tuttu, että se noudattaa MNIST:n tavoin samaista rakennetta: 28x28 pikselin harmaasävykuvat, joissa on vaatekappaleita luokiteltavana. Luokkia on 10 ja kyseessä on *multi-class, single-label classification* -ongelma. Datasetti löytyy suoraan `torchvision.datasets`-moduulista jaettuna koulutus- ja testidataseteiksi. Niissä on yhteensä 70 000 kuvaa (60 000 koulutukseen, 10 000 testaukseen). Tämä kaikki on hyvin, hyvin tuttua MNIST:stä. Se, mikä muuttuu, on että kuvat ovat vaikeampia tulkita kuin numerot (sandaali vs. lenkkari on vaikeampi erottaa kuin 3 vs. 7). Jos käytät samaa 784-256-128-10 -arkkitehtuuria kuin aiempi MLP, ja Sigmoid-kerroksia, koulutus kestää jotakuinkin yhtä kauan kuin edellinenkin. Saavutettu tarkkuus tulee olemaan heikompi (n. 85 %) sadalla epookilla. 

    Huomaa, että *"oikea vastaus"* on vähemmän merkityksellinen kuin sinun oppimismatka ja ymmärryksen syventyminen. Kokeile erilaisia arkkitehtuureita, optimointimenetelmiä ja hyperparametreja. Dokumentoi oppimiskokemuksesi. Oikean vastauksen sinulle antaa tuore kielimalli hetkessä, tai voit jopa kaivaa sen netistä, kuten [PyTorch Learn the Basics: Datasets & Dataloaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) tai AI and ML for Coders in PyTorch -kirjan [repositoriosta](https://github.com/lmoroney/PyTorch-Book-FIles/blob/main/Chapter02/PyTorchChapter2.ipynb). Huijaamalla huijaat lähinnä vain itseäsi.

    Älä siis huijaa vaan ota ensimmäisen viikon `110_first_model.py`-tiedosto esimerkiksi. Voit käyttää pohjana `400_fashion_mnist.py`-tiedostoa tai luoda kokonaan oman Notebookisi. 
    
    * Aloita pienestä mallista ja kasvata sitä, kunnes löydät ylisovittamisen rajan.
    * Käytä Tensorboardia prosessin seurantaan ja dokumentointiin.
    * Tallenna malli seuraavaa tehtävää varten!

    Kun tallennat arvot, tallenna mukana myös mallin arkkitehtuuriin liittyvät tiedot, joiden avulla saat alustettua täysin saman mallin uusiksi.

    !!! tip "Mallin tallennus"

        Tallenna malli PyTorchin `torch.save()`-funktiolla. Tallenna mukaan myös mallin arkkitehtuuriin liittyvät tiedot, kuten kerrosten lukumäärä ja solmujen lukumäärä kussakin kerroksessa. Näin voit alustaa täysin saman mallin uudelleen lataamisen yhteydessä. Tämän voi tehdä monin eri tavoin, mutta alla on yksi esimerkki:

        ```python
        model_checkpoint = {
            'input_size': 784,
            'output_size': 10,
            'hidden_layers': [256, 128, 64],
            'state_dict': model.state_dict()
        }

        torch.save(model_checkpoint, 'mnist_mlp_checkpoint.pth')
        ```

        Jos haluat haastaa itseäsi, kokeile kääntää malli TorchScript-muotoon `torch.jit.script()`-funktiolla. Löydät tähän hyvän esimerkin **Hands-On Machine Learning with Scikit-Learn and PyTorch** -kirjan luvun 10 lopusta otsikon *"Compiling and Optimizing a PyTorch Model"* alta.

        !!! tip "Relu vs. Sigmoid"

            Kannattaa kokeilla ainakin ReLU-aktivointia Sigmoidin sijasta, minkä jo itsessään pitäisi nostaa tarkkuutta muutaman prosenttiyksikön verran.

!!! question "Tehtävä: Lataa Fashion MNIST -malli"

    Käytä edellisen tehtävän tallennettua mallia ennustamaan luokkia Fashion MNIST -datasetin testidatalle. Tämän Notebookin kanssa sinulla ei ole sitä ongelmaa, että vahingossa ajaisit useita minuutteja vievän solun uudestaan. Toisin sanoen tässä sinun on helppo harjoitella seuraavia asioita:

    * Mallin lataaminen tiedostosta.
    * Mallin käyttäminen ennustamiseen.
    * Ennusteiden evaluointi sopivilla mittareilla (kuten top-k accuracy).
    * Output-kerroksen numeroiden tutkiminen
        * Aktivaatio on None, joten ulostulo on raakaa logit-arvoa.
        * Tulosta arvot sinällään ja todennäköisyydet (softmaxin avulla).

    Voit käyttää pohjana `401_fashion_mnist_eval.py`-tiedostoa tai luoda kokonaan oman Notebookisi.

!!! question "Tehtävä: Tutustu aktivointifunktioihin"

    Aja Marimo Notebook `413_activation_functions.ipynb` ja tutustu eri aktivointifunktioihin.

## Lähteet

[^mathforai]: Nelson, H. *Essential Math for AI*. O'Reilly Media. 2023.
[^dlwithpython]: Watson, M & Chollet, F. *Deep Learning with Python, Third Edition*. Manning. 2025.