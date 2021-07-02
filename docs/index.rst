.. Bagua API Documentation documentation master file, created by
   sphinx-quickstart on Thu Jun 10 16:09:03 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. .. toctree::
..    :maxdepth: 4
..    :caption: API Documents:

Bagua
======

This website contains Bagua API documentation. See `tutorials <https://baguasys.github.io/tutorials/>`_ if you need step by step instructions on how to use Bagua.


.. toctree::
   :titlesonly:
   :caption: API Documents

   {% for page in pages %}
   {% if page.top_level_object and page.display %}
   {{ page.include_path }}
   {% endif %}
   {% endfor %}
