{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :special-members: __init__
   :exclude-members: __weakref__, set_fit_request, set_predict_request, set_score_request

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
      {%- if item not in inherited_members and not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
