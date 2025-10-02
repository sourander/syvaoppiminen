---
priority: 110
---

# Syväoppiminen ja FC-verkot

TODO! Tähän tulee ainakin seuraavat asiat:

* Case: Fully Connected -verkko (MLP)
* Termistö:
    * Painot (weights)
    * Bias
    * Aktivaatiofunktio (activation function)
    * Loss-funktio (loss function)
    * Optimointi (optimization)
    * Epoch
    * Batch
    * Learning rate
    * Tämä kaikki viittaa ML-perusteet kurssilta saatuun pohjaan.


## Syväoppiminen

### Määritelmä

Määritelmän osalta Understanding Deep Learning kirjan luvun 3 alku on hyvä:

> "The last chapter described shallow neural networks, which have a single hidden layer. This chapter introduces deep neural networks, which have more than one hidden layer. With ReLU activation functions, both shallow and deep networks describe piecewise linear mappings from input to output." [^udlbook]

Prince toteaa, että matalan neuroverkkojen kyky kuvata monimutkaisia funktioita kasvaa piilokerroksen neuronien määrän lisääntyessä. Riittävän suurella neuronimäärällä matalat verkot pystyvät mallintamaan mielivaltaisen monimutkaisia funktioita. Käytännössä tämä on kuitenkin usein mahdotonta, sillä tarvittava neuronien määrä voi kasvaa kohtuuttoman suureksi. [^udlbook]

Syvät neuroverkot tarjoavat tähän ratkaisun: ne pystyvät tuottamaan huomattavasti enemmän lineaarisia alueita kuin matalat verkot samalla parametrimäärällä. Kerrosten määrän ja niiden neuronien määrä on siis jotakin, mitä pitää optimoida mallia suunniteltaessa. Palaamme tähän kohta tässä samassa luvussa.

### Yleiskatsaus

Täysin yhdistetyt kerrokset *(engl. fully connected layers)* ovat syväoppimisen peruskomponentteja. Niissä jokainen neuroni on yhteydessä kaikkiin edellisen kerroksen neuroneihin. Tämä mahdollistaa monimutkaisempien suhteiden oppimisen syötteiden ja ulostulojen välillä. Tosielämän mallit ovat 2020-luvulla siirtyneet yhä enemmän erilaisiin konvoluutio- ja toistoverkkoihin, mutta FC-kerrokset ovat edelleen keskeisiä monissa arkkitehtuureissa. Tässä luvussa keskitymme verkkoihin, joissa on pelkkiä FC-kerroksia.

Konsepti on helppo ja tulee toivon mukaan selväksi seuraavaa kuvaa katsomalla. Huomaat, että olet toteuttanut näitä verkkoja jo edellisen luvun TensorFlow Playground -tehtävässä.

Kuvassa kerroksen yksi neuronit ovat $h_1$, $h_2$ ja $h_3$. Kunkin niiden tuloste päätyy seuraavan kerroksen kunkin neuronin syötteeksi.

![](../images/110_DeepTwoLayer.svg)

**Kuva 1:** *Kaksi kerroksinen syväverkko, jossa on kaksi piilotettua kerrosta, joissa kussakin on kolme neuronia. Jokainen piilotettu kerros on täysin yhdistetty (fully connected) edelliseen kerrokseen.*

### Laskutoimitukset

Selvyyden vuoksi käydään läpi, miten verkko toimii. Oletetaan, että syötevektori on $x = [x_1, x_2]$. Ensimmäisen piilotetun kerroksen neuronit laskevat seuraavasti: 

$$
\begin{align*}
h_1 &= a(w_{11} x_1 + w_{12} x_2 + b_1) \\
h_2 &= a(w_{21} x_1 + w_{22} x_2 + b_2) \\
h_3 &= a(w_{31} x_1 + w_{32} x_2 + b_3)
\end{align*}
$$

missä $w_{ij}$ ovat painot ja $b_i$ on bias-termi. Funktio $a$ on aktivointifunktio ReLU. Toinen kerros on monimutkaisempi, koska siinä on kolme syötettä ja kolme vastaanottavaan neuronia, joten painoja tulee olemaan $3 \times 3 = 9$. Toisen kerroksen neuronit laskevat seuraavasti:

$$
\begin{align*}
h_4 &= a(w_{41} h_1 + w_{42} h_2 + w_{43} h_3 + b_4) \\
h_5 &= a(w_{51} h_1 + w_{52} h_2 + w_{53} h_3 + b_5) \\
h_6 &= a(w_{61} h_1 + w_{62} h_2 + w_{63} h_3 + b_6)
\end{align*}
$$

Nämä merkinnät alkavat olla kohtalaisen sekavia, joten on parempi käyttää vektori- ja matriisimerkintöjä. Kummankin kerroksen ja ulostulon laskenta voidaan ilmaista seuraavasti:

$$
\begin{align*}
h^{(1)} &= a(W^{(1)} x + b^{(1)}) \\
h^{(2)} &= a(W^{(2)} h^{(1)} + b^{(2)}) \\
y &= W^{(3)} h^{(2)} + b^{(3)}
\end{align*}
$$

Yllä olevassa kaavassa $y$ on laskettu ilman aktivointifunktiota. Tämä tekee mallista regressiomallin, joka sopii hyvin jatkuvien arvojen ennustamiseen. Jos mallia halutaan käyttää binääriseen luokitteluun, siihen lisätään sigmoid-aktivointifunktio. Huomaa, että tämä on kertausta Johdatus koneoppimiseen -kurssin logistisesta regressiosta.

Kurssin aikana tulemme käyttämään PyTorch-kirjastoa, joka hoitaa paljon laskuja puolestamme. Tässä on jo esimakua siitä, kuinka yllä olevat laskut voidaan toteuttaa PyTorchilla:

```python
# Esimerkin vuoksi W1 voisi näyttää tältä
W1 = torch.tensor(
    [
    [0.11, 0.12],
    [0.21, 0.22],
    [0.31, 0.32]
], dtype=torch.float32)

# Lasketaan ensimmäinen kerros
h1 = torch.relu(torch.matmul(W1, x) + b1)

# Lasketaan toinen kerros
h2 = torch.relu(torch.matmul(W2, h1) + b2)

# Lasketaan ulostulo (ilman aktivointifunktiota)
y = torch.matmul(W3, h2) + b3
```

Huomaa, että tässä on kyseessä pelkkä *inferenssi* eli ennustaminen. Koko mallin kouluttaminen vaatii vielä paljon enemmän koodia, ja tämä esitellään kurssilla myöhemmin.

### Hyperparametrit

Syväverkkojen suunnittelussa on useita hyperparametreja. Hyperparametrit ovat malliin liittyvät asetukset, jotka valitetaan ennen sen kouluttamista – eli niitä ei siis opita koulutusvaiheessa. Tässä luvussa keskitymme vain niihin hyperparametreihin, jotka liittyvät FC-verkon kokoon:

* Kerrosten määrä $K$
* Neuronien määrä kussakin kerroksessa $D_k$
  
Tutustumme myös muihin hyperparametreihin kurssin edetessä. Hyperparametrien *oikeita arvoja* ei voi yksinkertaisesti tarkistaa jostakin Maolin taulukkokirjasta. Ne on löydettävä kokeilemalla.

![](../images/110_DeepKLayer.svg)

**Kuva 2:** *Syväverkko, jossa on $K$ piilotettua kerrosta, joissa kussakin on $D_k$ neuronia. Jokainen piilotettu kerros on täysin yhdistetty (fully connected) edelliseen kerrokseen. Tähän kuvaan on piirretty mukaan myös vakiotermit (bias) $b_k$, jotka ovat $D_k$-ulotteisia vektoreita.*

Huomaa, että jos meidän *budjetti* GPU-muistille sallii vain $N = 1000$ painoa, voimme valita esimerkiksi luoda $K=2$ kerrosta, joissa kummassakin $D_k = 500$ neuronia. Tai voimme tehdä $K=5$ kerrosta, joissa kussakin on $D_k = 200$ neuronia. Tai voimme luoda suppilon, jossa ensimmäisessä kerroksessa on $D_1 = 400$ neuronia, toisessa $D_2 = 300$, kolmannessa $D_3 = 200$ ja neljännessä $D_4 = 100$. Kaikki nämä vaihtoehdot käyttävät saman verran muistia, mutta niillä on erilaiset kyvyt oppia erilaisia funktioita. Jos tutkit vanhoja malleja, huomaat, että suppilo oli ennen hyvinkin suosittu arkkitehtuuri. Nykyään on tavallisempaa käyttää saman kokoisia kerroksia. Géron antaa nyrkkisäännön, että paremman hyödyn saa tyypillisesti lisäämällä kerrosten määrää kuin neuronien määrää kerroksessa. [^handson-tf]

> "A typical neural network for MNIST might have 3 hidden layers, the first with 300 neurons, the second with 200, and the third with 100. However, this practice has been largely abandoned because it seems that using the same number of neurons in all hidden layers performs just as well in most cases, or even better; plus, there is only one hyperparameter to tune, instead of one per layer. That said, depending on the dataset, it can sometimes help to make the first hidden layer bigger than the others." [^handson-tf]

## MNIST ja FC-verkko

TODO! Selitä tässä MNIST-tehtävään liittyvä ongelma. Tästä on alempana *"Aja koodi ja tutki, mitä tapahtuu"*-tason tehtävä.

Koulutuksen tulokset eri raudalla:

| Rauta             | Aika/epookki | Aika/koko koulutus | Lopputarkkuus (val) |
| ----------------- | ------------ | ------------------ | ------------------- |
| MacBook Pro (CPU) | 2.9s         | 4.8min             | 95.35 %             |
| MacBook Pro (MPS) | 3.8s         | 6.2min             | 95.18 %             |
| PC (CPU)          | TODO         | TODO               | TODO                |
| PC (CUDA)         | TODO         | TODO               | TODO                |

![](../images/110_mlp_mnist_training_loss_and_acc.png)

**Kuva 4:** *MNIST-datalla koulutetun mallin tarkkuus (accuracy) ja häviö (loss) koulutuksen aikana epookki epookilta.*

### Termistöä

TODO


## Tehtävät

!!! question "Tehtävä: UDLbook Deep"

    Lue Understanding Deep Learning kirjasta vähintään luvu 4.1 *"Composing neural networks"* sekä 4.2 *"From composing networks to deep networks"*. Kirjoita itsellesi lyhyt yhteenveto aiheesta omin sanoinesi, jotta ymmärrät asian.

    Tutustu myös UDL-kirjan kylkiäisenä tuleviin [Interactive Figures](https://udlbook.github.io/udlfigures/)-työkaluihin. Erityisesti *Concatenating networks* sekä *Deep network computation*-kuvaajiin.

!!! question "Tehtävä: Valitse kehitysympäristösi"

    Valitse itsellesi sopiva kehitysympäristö, jossa aiot tehdä kurssin harjoitukset. Vaihtoehtoja on useita:

    * Lokaali kone:
        * `uv`-ympäristö ja VS Code Notebooks (tätä opettaja käyttää macOS:llä ja Ubuntussa)
        * `docker`-ympäristö ja Jupyter Lab (Windows + CUDA GPU suositus)
    * Jupyter Hub (DC-labran ylläpitämä)
    * Coder (DC-labran ylläpitämä)
    * Google Colab
    * Joku muu pilvipalvelu, jossa on GPU

    HUOM! Opettaja ei voi realistisesti kokeilla kaikkia vaihtoehtoja, jotka syntyvät `("Win", "Mac", "Linux") x ("uv", "docker", "jupyterhub", "colab")` -ristikkona. Valitse siis sellainen, joka sinulle on tuttu tai jonka opit helposti. Opettaja tarjoaa tukea, mutta älä odota, että sinulle annetaan tasan yksi koodirimpsu, jolla kaikki toimii. Hallitse omat ympäristösi!

!!! question "Tehtävä: Aja MNIST MLP koodi"

    Koodi löytyy Notebookista `notebooks/nb/100/110_first_model.ipynb` tämän kurssimateriaalin repositoriota eli [gh:sourander/syvaoppiminen](https://github.com/sourander/syvaoppiminen).

    Lataa Notebook koneellesi ja testaa ympäristösi toimivuus.
    
    Aja se joko omalla koneellasi tai Google Colabissa. Varmista, että saat mallin koulutettua ja tarkkuuden nousemaan vähintään 97%:iin.

## Lähteet

[^udlbook]: Prince, S. *Understanding Deep Learning*. The MIT Press. 2023. https://udlbook.github.io/udlbook/
[^handson-tf]: Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition*. O'Reilly Media. 2022.