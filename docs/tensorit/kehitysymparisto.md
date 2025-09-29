---
priority: 200
---

# Kehitysympäristö

Tätä kurssia voi ajaa omalla koneella tai pilvessä. Molemmissa on omat etunsa ja haittansa.

Tässä tulee olemaan lyhyt ohjeistus seuraaviin:

* DC Labran Jupyter Hub (GPU-tuki)
* Lokaali setup ilman Dockeria:
    * Jupyter Lab
    * CUDA/rocM
    * PyTorch asennus
* Lokaali setup Dockerilla:
    * Docker Desktop
    * Docker image (PyTorch + Jupyter Lab)
    * nvidia-container-toolkit (tarvittaessa)

En aio ylläpitää asennusohjeita kaikille mahdollisille alustoille, joten tähän tulee lähinnä pointtereita ja linkkejä hyviin ohjeisiin sekä yleipäteviä ohjeita. Opiskelijalta oletetaan itsenäisyyttä ja kyky selvittää omat asennusongelmansa.

Lisäksi: tässä pitää olla jokin "Hello World"-esimerkki, jonka opiskelija voi kopioida Notebookiin ja tsekata, että sekä CPU että GPU toimivat.