---
priority: 500
---

# Konvoluutioverkot (CNN)

## Perusteet

### Motivaatio

Olet tutustunut kurssilla FCNN-verkkoihin, ja niiden rajat alkoivat löytyä Cifar10-datasetin kohdalla. Edellisen luvun tehtävässä koulutit FCNN-verkon – kenties arkkitehtuurilla `3072-1024-512-10` –, ja pääsit noin 55% tarkkuuteen. Tutustuessasi wikipedian [Cifar10](https://en.wikipedia.org/wiki/CIFAR-10)-sivuun huomasit, että jo 2010-luvun alkupuolella verkot kykenivät yli 95 % tarkkuuteen. On hyvä muistaa, että 95 % on jo merkittävän suuri tarkkuus. Graham lainaa Karpathyä, että: *"For comparison, human performance on CIFAR-10 is estimated to be 6%."* [^fractionalmp] Lukema on siis *error rate*, ei *accuracy*. Ihmiseen kun vertaa, niin jo vuonna 2014 Grahamin **Fractional Max-Pooling** -malli saavutti huimat tulokset: *"we obtained test errors of 4.50% (1 test), 3.67% (12 tests) and 3.47% (100 tests)"* [^fractionalmp]. Konvoluutioverkot mahdollistavat siis huomattavasti tehokkaamman tavan käsitellä kuvia. Ja mikä oli Grahamin mallin parametrien määrä? **74 miljoonaa** parametria, jos laskin oikein (jos `filter_growth_rate = 160`). Tosin paperissa mainitaan myös 12M parametria käyttänyt malli (`filter_growth_rate = 64`).

Miten tämä on mahdollista? Lienee selvää, että 2010-luvun alussa ratkaisu tuskin oli kasvattaa verkkoa ilman arkkitehtuurimuutoksia. Alla on taulukossa esiteltynä `3072-1024-512-10` FCNN-verkon parametrien lukumäärä. 


| Layer      | Shape                    | Count         |
| ---------- | ------------------------ | ------------- |
| fc1.weight | torch.Size([1024, 3072]) | 3,145,728     |
| fc1.bias   | torch.Size([1024])       | 1,024         |
| fc2.weight | torch.Size([512, 1024])  | 524,288       |
| fc2.bias   | torch.Size([512])        | 512           |
| fc3.weight | torch.Size([10, 512])    | 5,120         |
| fc3.bias   | torch.Size([10])         | 10            |
| **Total**  |                          | **3,676,682** |

Vuonna 2014 olisi ollut mahdollista käyttää esimerkiksi GeForce GTX TITAN -korttia, jossa on 6 GB muistia. Muistiin mahtuisi $\frac{6 \times 1024^3}{4} \approx 1600 \text{M}$ parametria (per BATCH). Jos *batch size* on 32, niin silloin käytössä on $\frac{1600}{32} \approx 50 \text{M}$ parametria. Tämä on yhä reilusti enemmän kuin edellä esitellyssä verkossa. Koulutusvaiheessa muistiin pitää mahtua myös aktivoinnit, gradientit ja optimointiin liittyvät muuttujat. Muistiin mahtuisi joka tapauksessa jo 2014 vuoden raudalla hyvinkin suuri malli, mutta olet varmasti kokeillut tätä ratkaisua itsekin: verkon kokoa kasvattamalla ei päästä kovin pitkälle. Mikä siis avuksi? Historiasta löytyy vastaus: konvoluutioverkot (Convolutional Neural Networks, CNN). Kuten yllä on todettu, noin $74 \text{M}$ parametria riitti Grahamin mallissa jo 3,47 % virheeseen CIFAR-10 datasetin kanssa.

!!! note "Entä suurempi input?"

    VGG-16-konvoluutioverkossa käytetään tyypillisesti kuvia koossa 224×224×3 (RGB, 3 kanavaa). Dataset on nimeltään ImageNet, jossa on 1000 eri luokkaa ja miljoonia kuvia. Koko verkossa on pelkästään 138M parametria.

    Kuinka olisi FCNN-verkon laita, jos input on `224x224x3` ja ensimmäinen piilotettu kerros `4096` neuronia? Input olisi siis `150,528`-ulotteinen vektori. Tällöin ==pelkästään ensimmäisen== piilotetun kerrokset olisivat parametrimäärältään:

    $$
    150,528 \times 4096 \approx 617 \text{M}
    $$

### Lyhyt historia

* **1980**: Konvoluutioverkkojen juuret ulottuvat 1980-luvulle, jolloin Kunihiko Fukushima esitteli Neocognitron-mallin, josta polveutuvat myöhemmät konvoluutioverkot. [^neocognition] 
* **1994**:MNIST-dataset ja LeNet-4.
* **1998**: LeNet-5, tunnetuin näistä LeNet-X -malleista. ~60k parametria [^lenet5].
* **2012**: AlexNet, merkittävä edistysaskel syvien konvoluutioverkkojen koulutuksessa, joka voitti ImageNet-kilpailun ylivoimaisesti. [^alexnet].
* **2015**: VGG-16, syvä konvoluutioverkko, joka käytti nimensä mukaisesti 16 kerrosta. ~138 M parametria. [^vgg16] [^vgg16neurohive].
* **2015**: ResNet, esitteli "residual connections", jotka mahdollistivat erittäin syvien verkkojen koulutuksen. ~19 M parametria. [^resnet] [^resnetmedium].
* **2018**: DenseNet, joka käytti tiheitä yhteyksiä kerrosten välillä parantaakseen tiedonsiirtoa ja vähentääkseen gradientin katoamista. ~28 M parametria. [^densenet].

## Käytäntö

### Visual Feature Descriptors

TODO: Lyhyt katsaus vanhoihin feature descriptor -menetelmiin, kuten SIFT ja HOG.

### Konvoluutioverkkojen rakenne

Aiemmasta opitusta on hyötyä, sillä konvoluutioverkkojen *head* eli viimeiset kerrokset ovat tuttuja Dense/FC-kerroksia. Alkuosa, eli *body*, sisältää uudenlaisia kerroksia: **konvoluutiokerros** (*convolutional*) ja **koontikerros** (*pooling*).

#### Konvoluutiokerros

Lue tämä: [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way](https://medium.com/data-science/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

#### Koontikerros

TODO.

## Case Study: Fractional Max-Pooling (Graham, 2014)

Tässä osiossa syvennytään Benjamin Grahamin vuonna 2014 esittelemään **Fractional Max-Pooling** -arkkitehtuuriin, joka saavutti aikanaan poikkeuksellisen alle 4 % virheasteen CIFAR-10-datasetillä. Malli on erinomainen esimerkki siitä, kuinka konvoluutioverkkojen suunnittelussa voidaan poiketa valtavirran konventioista ja saavuttaa silti huipputuloksia. Motivaatio arkkitehtuurin valinnalle case studyyn on juurikin tämä: älä lukitse itseäsi ajattelemaan, että on vain yksi tapa rakentaa konvoluutioverkkoja tai muitakaan neuroverkkoja.

Case studyn koodi on toteutettu `501_fractional_max_pooling.py`-tiedostossa. Tähän osioon liittyy osion ensimmäinen tehtävä. Lue se alta.

#### Arkkitehtuurin filosofia ja suotimien kasvu

Tyypillisesti konvoluutioverkoissa, kuten VGG:ssä tai ResNetissä, kanavien määrä kaksinkertaistetaan tietyin väliajoin (esim. 64 $\to$ 128). Grahamin mallissa lähestymistapa on kuitenkin erilainen: kanavien määrä kasvaa lineaarisesti kaavalla $160 \times n$, missä $n$ on kerroksen järjestysnumero. Tämä lineaarinen kasvu johtaa verkon syventyessä massiiviseen parametrimäärään. Kun perinteiset verkot saattavat pysytellä kymmenissä miljoonissa parametreissa, Grahamin malli CIFAR-10:lle sisältää noin 75 miljoonaa parametria. Tämä osoittaa, että suuri parametrimäärä ei välttämättä johda ylikouluttumiseen, kunhan verkon muut rakenteelliset valinnat tukevat oppimista.

#### Fractional Max-Pooling -mekanismi

Mallin keskeinen innovaatio on nimensä mukainen Fractional Max-Pooling. Perinteinen $2\times2$ max-pooling puolittaa kuvan koon jokaisella askeleella, mikä rajoittaa verkon syvyyttä pienillä kuvilla. Grahamin ratkaisussa skaalauskerroin on $\sqrt[3]{2}$ (noin 1,26). Tämä maltillisempi koon pienentäminen mahdollistaa huomattavasti syvemmät verkot ilman, että piirrekartat kutistuvat liian nopeasti $1\times1$-pikselin kokoon.

Lisäksi menetelmä hyödyntää satunnaisuutta. Pooling-alueet voidaan valita joko limittäin (overlapping) tai erillisinä (disjoint), ja niiden sijainti arvotaan. Tämä tuo verkkoon sisäänrakennettua regularisointia, sillä verkko ei voi luottaa siihen, että tietyt piirteet löytyvät aina täsmälleen samasta kohdasta suhteessa pooling-ruudukkoon.

#### Moderni "Head" -rakenne

Verkon loppuosa poikkeaa myös perinteisestä. Sen sijaan, että piirrekartat litistettäisiin (flatten) ja syötettäisiin tiheille (Dense/Linear) kerroksille, malli käyttää $1\times1$ konvoluutiota (C1). Tämä kerros projisoi piirteet suoraan luokkien lukumäärää vastaavaksi vektoriksi. Tämä tekniikka vähentää parametrien määrää verkon loppupäässä ja on nykyään yleinen käytäntö ns. täysin konvoluutiopohjaisissa verkoissa (Fully Convolutional Networks).

#### Regularisointi ja koulutuksen erikoisuudet

Koska malli on valtava suhteessa datasetin kokoon, regularisointi on kriittistä. Mallissa sovelletaan "kasvavaa dropoutia": ensimmäisissä piilotetuissa kerroksissa dropout on 0 %, ja se kasvaa lineaarisesti 50 %:iin verkon loppupäätä kohden.

Toinen tekninen erikoisuus liittyy verkon syötteen kokoon. Koska pooling-suhde on murtoluku, verkon vaatima syötekoko ei ole triviaali laskea. Käytännössä haluttu output-koko päätetään ensin, ja vaadittu input-koko lasketaan "takaperin" kertomalla kokoa skaalauskertoimella jokaisen kerroksen kohdalla. Tämä johtaa usein siihen, että alkuperäisiä kuvia on pehmustettava (padding) runsaasti.

#### Inferenssi: Verkko on itsessään ensemble

Pooling-vaiheen satunnaisuudesta johtuen saman kuvan ajaminen verkon läpi useaan kertaan tuottaa hieman erilaisia ennusteita. Tätä ominaisuutta hyödynnetään testausvaiheessa (inference). Lopullinen luokitus ei perustu yhteen ajokertaan, vaan usean (esim. 12) ajokerran keskiarvoon (Model Averaging). Tämä toimii ikään kuin "köyhän miehen ensemble-menetelmänä", parantaen luotettavuutta ilman tarvetta kouluttaa useita erillisiä verkkoja. Tätä ei ole toteutettu kurssin koodissa, mutta voit halutessasi kokeilla tätä itse. Se hoituisi jotakuinkin näin:

```python
outputs = [model(image) for _ in range(12)]   # Run 12 times
avg_output = torch.stack(outputs).mean(dim=0) # Average predictions
```

## Tehtävät

!!! question "Tehtävä: Tutustuminen Fractional Max-Pooling -toteutukseen"

    Avaa tiedosto `notebooks/nb/500/501_fractional_max_pooling.py`. Tämä on teknisesti haastava toteutus, joka poikkeaa kurssin aiemmista malleista. Tutki koodia ja pohdi oppimispäiväkirjassasi seuraavia kysymyksiä:

    *   **Käänteinen koon laskenta:** Miten `get_fmp_sizes`-funktio toimii? Miksi verkon kerrosten koot lasketaan "lopusta alkuun" (output $\to$ input)?
    *   **Dynaaminen padding:** Toteutuksessa käytetään dynaamista paddingia (`pad_total`). Miksi tämä on välttämätöntä juuri tässä arkkitehtuurissa, kun taas esimerkiksi VGG-verkossa pärjätään kiinteällä paddingilla?
    *   **Verkon "häntä" (Tail):** Miten `FMPNet`-luokan `forward`-metodin loppuosa eroaa perinteisestä `nn.Linear`-kerroksesta? Miksi tässä on käytetty $1\times1$ konvoluutiota (`convC1`)? Onko kenties niin, että matemaattisesti $1\times1$ konvoluutio $1\times1$ kokoisella spatiaalisella kartalla on identtisen täysin kytketyn kerroksen kanssa?
    *   **Ensemble-ennustaminen:** Miten mallin ennusteet lasketaan testausvaiheessa? Miksi sama kuva syötetään verkolle useita kertoja? Eikö neuroverkko olekaan deterministinen? Miksi minun toteutus toimii, vaikka siinä ei syötetä kuin kerran?
    *   **Suotimien kasvu:** Miksi `filter_growth_rate` on toteutettu siten, että kanavien määrä kasvaa lineaarisesti ($k \times n$)? Tähän vastataan alkuperäisessä paperissa.
  
    P.S. En voi suositella tämän mallin ajamista CPU-raudalla.

!!! question "Tehtävä: FMP ja MNIST"

    Toteuta tiedostoon `notebooks/nb/500/502_fractional_max_pool_MNIST.py` versio Grahamin mallista, joka on sovitettu MNIST-datasetille.

    Alkuperäisessä paperissa (Graham, 2014) on määritelty MNIST:lle kevyempi arkkitehtuuri kuin CIFAR-10:lle. Etsi paperista (tai kokeile itse) sopivat parametrit ja muokkaa koodia seuraavasti:

    *   **Dataset:** Vaihda CIFAR-10 $\to$ MNIST. Huomioi, että MNIST on mustavalkoinen (1 kanava), kun taas CIFAR-10 on värillinen (3 kanavaa).
    *   **Arkkitehtuuri:** Paperin mukaan MNIST-mallissa käytetään eri määrää kerroksia ja eri kasvukerrointa.
        *   CIFAR-10-mallissa kerroksia oli 12 ja kasvukerroin suuri.
        *   MNIST-mallille riittää vähempi määrä kerroksia ja pienempi kasvukerroin.
        *   Myös skaalauskerroin $\alpha$ voi olla eri (esim. $\sqrt{2}$ vs $\sqrt[3]{2}$), jotta kuva kutistuu sopivasti 28x28-koosta.
    *   **Tavoite:** Kouluta malli ja vertaa saavuttamaasi tarkkuutta
  
    P.S. Voit kokeilla, kauan mallin koulutus kestää GPU vs. CPU. Jos haluat säästää aikaa, selvitä 10 epookkiin kuluva aika ja skaalaa se haluamaasi epookkimäärään.

!!! question "Tehtävä: LeNet ja MNIST"

    Toteuta tiedostoon `notebooks/nb/500/503_lenet_MNIST.py` LeNet-5 -arkkitehtuuri MNIST-datasetille. Vertaile sen suorituskykyä toteuttamaasi  Grahamin Fractional Max-Pooling -malliin. Saat itse valita, toteutatko mahdollisimman orjallisesti alkuperäisen mallin vai Adrian Rosebrockin tulkitseman modernisoidun version. 
    
    **Vaihtoehto 1: Orjallinen LeNet-5**
    
    Mallin arkkitehtuuri löytyy LeCunin alkuperäisestä paperista [^lenet5]. Jos haluat olla uskollisempi alkuperäiselle arkkitehtuurille, voit käyttää Sigmoid-aktivointia. Tarkempi toteutus on $1.7159 \times \tanh(\frac{2}{3}x)$.

    Jos sinua hämmentää 32x32 syötekoko MNIST:lle, voit lisätä kuviin 2 pikselin pehmusteen (padding) joka reunalle, jolloin kuvat ovat 32x32-kokoisia. Se liittyy paperin lauseeseen: *"In the first version, the images were centered in a 28 x 28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field In some instances, this 28x28 field was ex tended to 32x32 with background pixels"*.

    **Vaihtoehto 2: Modernisoitu LeNet-5**

    Modernisoitu versio on Adrian Rosebrockin blogipostauksesta [LeNet – Convolutional Neural Network in Python](https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/) tai hänen kirjastaan *Deep Learning for Computer Vision with Python*. Tässä versiossa käytetään ReLU-aktivointia. Arkkitehtuuri on seuraava:
    
    | Layer Type | Output Size | Filter Size / Stride |
    | ---------- | ----------- | -------------------- |
    | Input      | 28x28x1     |                      |
    | Conv1      | 28x28x20    | 5x5 / K = 20         |
    | Act1       | 28x28x20    | ReLU                 |
    | Pool1      | 14x14x20    | 2x2 / S = 2          |
    | Conv2      | 14x14x50    | 5x5 / K = 50         |
    | Act2       | 14x14x50    | ReLU                 |
    | Pool2      | 7x7x50      | 2x2 / S = 2          |
    | FC1        | 500         |                      |
    | Act3       | 500         | ReLU                 |
    | FC2        | 10          |                      |
    | Softmax    | 10          |                      |

    

## Lähteet

[^neocognition]: Fukushima, K. *Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position*. Princeton. https://www.cs.princeton.edu/courses/archive/spr08/cos598B/Readings/Fukushima1980.pdf

[^fractionalmp]: Graham, B. *Fractional Max-Pooling*. University of Warwick. https://doi.org/10.48550/arXiv.1412.6071

[^lenet5]: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE, 86(11). http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

[^alexnet]: Krizhevsky, A., Sutskever, I., & Hinton, G. E. *ImageNet Classification with Deep Convolutional Neural Networks*. https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

[^vgg16]: Simonyan, K., & Zisserman, A. *Very Deep Convolutional Networks for Large-Scale Image Recognition*. https://arxiv.org/abs/1409.1556

[^vgg16neurohive]: Hassan, M. *VGG16 – Convolutional Network for Classification and Detection*. Neurohive. https://neurohive.io/en/popular-networks/vgg16/

[^resnet]: He, K., Zhang, X., Ren, S., & Sun, J. *Deep Residual Learning for Image Recognition*. https://arxiv.org/abs/1512.03385

[^resnetmedium]: Azeem. *Understanding ResNet Architecture: A Deep Dive into Residual Neural Network*. https://medium.com/@ibtedaazeem/understanding-resnet-architecture-a-deep-dive-into-residual-neural-network-2c792e6537a9

[^densenet]: Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. *Densely Connected Convolutional Networks*. https://arxiv.org/abs/1608.06993