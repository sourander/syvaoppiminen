---
priority: 420
---

# Kouluttamisen käytännöt

## Geronin vinkit

Tässä osiossa käydään läpi mallin kouluttamiseen liittyviä ==hyviä tai yleisiä käytäntöjä==. Jatkossa kurssilla edetään eteenpäin kytkeytyvistä, täysin kytketyistä verkoista (*engl. feedforward fully connected neural networks, multilayer perceptron*) kohti oudompia rakenteita, kuten konvoluutioverkkoja ja sekvensseihin soveltuvia rekursiivisia verkkoja. Ennen näitä on hyvä varmistaa, että päädyt johonkin johtopäätökseen siitä, millä oletusarvoilla voit lähteä liikenteeseen esimerkiksi projektityössäsi, mikäli tehtävään soveltuu MLP/FFNN-malli.

On oletus, että luet tässä yhteydessä kurssikirjan **Hands-On Machine Learning with Scikit-Learn and PyTorch**[^geronpytorch] luvun 10 sekä 11 [^geronpytorch]. Voi olla miellekästä selata myös erittäin kansantajuista kirjaa nimeltään **Machine Learning Yearning**[^mlyearning]. Toinen kurssikirja, **Understanding Deep Learning**, antaa hyvin taustaa erityisesti regularisointiin. Toivon mukaan olet jo valmiiksi edennyt luvun 8 loppuun asti kirjassa, sillä kurssin aiheet ovat tähän asti liittyneet pitkälti kirjan lukuihin 1–8. Alla on taulukko, joka on lainattu suoraan kurssikirjasta[^geronpytorch]. Se toimikoon hyvänä koosteena termeistä, jotka sinun tulisi osata selittää omin sanoin oppimispäiväkirjaasi.

| Hyperparameter          | Default value                                     |
| ----------------------- | ------------------------------------------------- |
| Kernel initialization   | He initialization                                 |
| Activation function     | ReLU if shallow; Swish if deep                    |
| Normalization           | None if shallow; batch-norm or layer-norm if deep |
| Regularization          | Early stopping; weight decay if needed            |
| Optimizer               | Nesterov accelerated gradient or Adam             |
| Learning rate scheduler | Performance scheduling or 1cycle                  |

**Taulukko 1.** Yleisiä hyperparametreja ja niiden oletusarvoja. Taulukko on englanniksi alkuperäislähteen mukaisesti. [^geronpytorch]

## Karpathyn vinkit

Kuten Karpathy toteaa artikkelissaan [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/), mallien koulutus vaatii kärsivällisyyttä, tarkkuutta ja systemaattista lähestymistapaa, koska valmiit kirjastot eivät poista virheiden riskiä. Onnistuminen perustuu datan tuntemiseen, yksinkertaisista malleista aloittamiseen ja vaiheittaiseen monimutkaistamiseen. Karpathyn vinkeissä on samoja piirteitä kuin [Yleiskatsaus](yleiskatsaus.md)-osion työnkulussa, mutta nämä sukeltavat lähemmäs varsinaista käytäntöä. Alla poimintaa artikkelista maistiaisena: [^karpathy]

* Tunne data kuin omat taskusi
* Luo baseline ja pipeline
* Pidä lego-palikat aluksi yksinkertaisina
* Ylisovita
* Regularisoi

## Tehtävät

!!! question "Tehtävä: Kouluttamisen käytännöt"

    Selitä tiiviisti alla listatut termit omin sanoin oppimispäiväkirjaasi. Kirjaa asiat ylös siten, että voit jatkossa käyttää sitä omana muistiinpanona, kun sinun pitää palata näihin aiheisiin. Voit käyttää apuna kurssimateriaaleja ja muita lähteitä, mutta kirjoita asiat omin sanoin. Mikäli koodin käyttö on tarpeen, voit käyttää sitä havainnollistamaan asiaa. Se on jopa suositeltavaa osassa termeissä.

    * Validation split (sinulle on training ja testing jo aiemmin tuttua tavaraa.) (ks. :two: luvut 5–7 ja 12)
    * Vanishing gradients (ks. :one: luku 11)
    * Regularisointi (dropout, L1 ja L2 weight decay) (ks. :one: ja :three: )
    * Error analysis (ks. :two: luku 14–19)

    Alla lähdevinkkejä:
    
    :one: **Hands-On Machine Learning with Scikit-Learn and PyTorch** -kirjaa eli virallista kurssikirjaa. Se löytyy koulun kirjastosta.

    :two: **Machine Learning Yearning** -kirjaa, joka on ilmainen e-kirja. Löytyy osoitteesta: [andrew-ng-machine-learning-yearning.pdf](https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf)

    :three: **Understanding Deep Learning** -kirja antaa ryhtiä teorialle. Regularisoinnista on kokonainen luku 9.

!!! question "Tehtävä: Piirrä multihead malli (Gaussian)"

    Avaa `420_gaussian_regression.py` ja tutustu sen koodiin, erityisesti mallin rakenteeseen. Vahvista osaamisesi piirtämällä malli valitsemallasi työkalulla (esim. Excalidraw). Malli on muotoa *multi-task learning* siinä mielessä, että se ennustaa kaksi lukua: mean ja variance. Mallilla on ns. *shared trunk* ja kaksi *headia*, mistä johtuu termi *multihead*.

    Kannattaa tutustua myös kurssikirjan luvun 10 lopulla olevaan "Building Nonsequential Models Using Custom Modules" -osioon, jossa on eri tavalla haarautuva verkko. Verkkossa osa inputista tuodaan "ohitusreittiä" piilotetun kerrosten ohi. 
    
    !!! tip "Skip connections / Residual connections"

        Ohitusreittejä kutsutaan englanniksi termeillä *skip connections* tai *residual connections*. Ne auttavat mallia oppimaan syvempiä verkkoja tehokkaammin, koska ne helpottavat gradientin kulkua taaksepäin verkon läpi. Törmännet näihin myöhemmin konvoluutioverkon ResNet arkkitehtuurissa.


## Lähteet

[^geronpytorch]: Géron, A. *Hands-On Machine Learning with Scikit-Learn and PyTorch*. O'Reilly. 2025.
[^mlyearning]: Ng, A. *Machine Learning Yearning*. 2018. https://info.deeplearning.ai/machine-learning-yearning-book
[^karpathy]: Karpathy, A. *A Recipe for Training Neural Networks*. 2019. https://karpathy.github.io/2019/04/25/recipe/
