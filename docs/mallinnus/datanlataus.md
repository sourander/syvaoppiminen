---
priority: 410
---

# Datan lataus

* **Datan lataus** (luku 4 kirjasta AI and ML for Coders in PyTorch)
    * ...ja luku 10 (Implementing Mini-Batch Gradient Descent Using DataLoaders) kirjasta Hands-On Machine Laerning with Scikit-Learn and PyTorch



## Tehtävät

!!! question "Tehtävä: CIFAR10 Datasetin plärääminen"

    Kouluta yksinkertainen neuroverkko [CIFAR10](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) -datasetillä. Datasetti koostuu 60 000 ==värikuvasta==. Kuvia on 6,000 kutakin luokkaa kohden eli luokkia on 10. Harjoittele seuraavia:

    * Lataa CIFAR10-datasetti PyTorchin `torchvision.datasets`-moduulista.
    * Tutki datan rakennetta (kuinka monta kuvaa, kuvan koko, värit, luokat).
    * Visualisoi yksi kuva
    * Visualisoi useampi kuva ruudukossa (grid). 
        * Bonus: Käyttäjä voi valita luokan.
    * Luo `DataLoader`, jolla voit iteroida datan läpi mini-batcheina.
        * Visualisoi yksi mini-batch ruudukossa.


!!! question "Tehtävä: CIFAR10 Malli"

    Kouluta myös yksinkertainen eteenpäin kytkeytyvä (feedforward) neuroverkko CIFAR10-datasetillä. Tulemme myöhemmin vertaamaan tätä suoritusta konvoluutioverkkoon (CNN).

    Alta saat jo osviittaa, mihin lukemiin tulet pääsemään ==seuraavissa luvuissa==. Nyt voit olla tyytyväinen, jos pääset noin 60 % tarkkuuteen (accuracy). Mikä mahtaa olla baseline, jotta olet arpaa parempi?

    | Paper title                                                                | Error rate | Accuracy | Year |
    | -------------------------------------------------------------------------- | ---------- | -------- | ---- |
    | Convolutional Deep Belief Networks on CIFAR-10                             | 21.1 %     | 78.9%    | 2010 |
    | Maxout Networks                                                            | 9.38 %     | 90.62%   | 2013 |
    | Wide Residual Networks                                                     | 4.0 %      | 96.0%    | 2016 |
    | ...                                                                        | ...        | ...      | ...  |
    | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | 0.5 %      | 99.5%    | 2021 |

    Taulukon lähde: [Wikipedia: CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10)
