"""
SMARTUR Context Encoder v3
Transforma el JSON del formulario React en vectores numéricos para el modelo RF.
Responsable de:
  - Mapeo ordinal de presupuesto y rango de edad
  - Multi-hot encoding de tipos de turismo
  - Features de match (distancia presupuesto-precio, overlap intereses-categorías)
"""

# Mapeo de categorías Yelp por tipo de turismo (reutilizado de fusion.py)
MAPEO_CATEGORIAS = {
    'naturaleza': ['Parks', 'Botanical Gardens', 'Hiking',
                   'Landmarks & Historical Buildings', 'Lakes'],
    'aventura':   ['Active Life', 'Hiking', 'Rafting',
                   'Mountain Biking', 'Tours'],
    'gastronomico': ['Restaurants', 'Food', 'Cafes',
                     'Traditional Mexican', 'Bakeries'],
    'cultural':   ['Museums', 'Art Galleries', 'Arts & Entertainment',
                   'Historical Tours', 'Festivals'],
    'rural':      ['Hotels', 'Bed & Breakfast', 'Campgrounds',
                   'Farm Stays', 'Guest Houses'],
}

TOURISM_TYPES = list(MAPEO_CATEGORIAS.keys())

BUDGET_MAP = {
    'bajo': 1,
    'medio': 2,
    'alto': 3,
    'premium': 4,
}

AGE_MAP = {
    '18-24': 1,
    '25-34': 2,
    '35-44': 3,
    '45-54': 4,
    '55+': 5,
}

# Mapeo de budget ordinal → price_level equivalente para calcular delta
BUDGET_TO_PRICE = {
    'bajo': 1,
    'medio': 2,
    'alto': 3,
    'premium': 4,
}


class ContextEncoder:
    """Codifica el contexto del turista en features numéricas."""

    def __init__(self):
        self.tourism_types = TOURISM_TYPES
        self.budget_map = BUDGET_MAP
        self.age_map = AGE_MAP

    def encode_user(self, context):
        """
        Convierte el dict de contexto del formulario React en features numéricas del usuario.

        Args:
            context (dict): JSON del formulario con campos como
                presupuesto_bucket, edad_range, tiposTurismo, etc.

        Returns:
            dict: Features numéricas del usuario:
                - user_budget (int 1-4)
                - user_age_range (int 1-5)
                - user_tur_* (0/1 por cada tipo de turismo)
                - user_requires_accessibility (0/1)
                - user_pref_outdoor (0/1)
        """
        if not context:
            return self._default_user_features()

        features = {}

        # Ordinal: presupuesto
        budget_raw = str(context.get('presupuesto_bucket', '')).lower().strip()
        features['user_budget'] = self.budget_map.get(budget_raw, 2)

        # Ordinal: rango de edad
        age_raw = str(context.get('edad_range', '')).strip()
        features['user_age_range'] = self.age_map.get(age_raw, 2)

        # Multi-hot: tipos de turismo
        tipos = context.get('tiposTurismo', [])
        if isinstance(tipos, str):
            tipos = [tipos]
        for t in self.tourism_types:
            features[f'user_tur_{t}'] = 1 if t in tipos else 0

        # Binarios de preferencia
        features['user_requires_accessibility'] = (
            1 if context.get('requiere_accesibilidad', False) else 0
        )
        features['user_pref_outdoor'] = (
            1 if context.get('pref_outdoor', False) else 0
        )

        return features

    def _default_user_features(self):
        """Features por defecto cuando no hay contexto (cold start de contexto)."""
        features = {
            'user_budget': 2,
            'user_age_range': 2,
            'user_requires_accessibility': 0,
            'user_pref_outdoor': 0,
        }
        for t in self.tourism_types:
            features[f'user_tur_{t}'] = 0
        return features

    def compute_match_features(self, user_features, biz_row):
        """
        Calcula features de interacción user × item.

        Args:
            user_features (dict): Output de encode_user()
            biz_row (pd.Series o dict): Fila del DataFrame de negocios

        Returns:
            dict: Features de match:
                - budget_delta: |user_budget - price_level|
                - interest_overlap: nº de categorías del negocio que coinciden
                  con los tiposTurismo del usuario
        """
        # Budget delta
        user_budget = user_features.get('user_budget', 2)
        price_level = int(biz_row.get('price_level', 2)) if 'price_level' in biz_row else 2
        budget_delta = abs(user_budget - price_level)

        # Interest overlap
        categories_str = str(biz_row.get('categories', ''))
        overlap = 0
        for t in self.tourism_types:
            if user_features.get(f'user_tur_{t}', 0) == 1:
                yelp_cats = MAPEO_CATEGORIAS.get(t, [])
                for cat in yelp_cats:
                    if cat.lower() in categories_str.lower():
                        overlap += 1
                        break  # contar max 1 por tipo de turismo

        return {
            'budget_delta': budget_delta,
            'interest_overlap': overlap,
        }

    def encode_pair(self, context, biz_row):
        """
        Construye el vector completo [User + Item + Interaction] para una pareja
        usuario-negocio. Usado por el RF para re-ranking.

        Args:
            context (dict): JSON del formulario React
            biz_row (pd.Series o dict): Fila del negocio

        Returns:
            dict: Vector completo de features combinadas
        """
        user_feats = self.encode_user(context)
        match_feats = self.compute_match_features(user_feats, biz_row)

        return {**user_feats, **match_feats}

    @property
    def user_feature_names(self):
        """Nombres de las features de usuario (para columnas del DataFrame)."""
        names = ['user_budget', 'user_age_range']
        names += [f'user_tur_{t}' for t in self.tourism_types]
        names += ['user_requires_accessibility', 'user_pref_outdoor']
        return names

    @property
    def match_feature_names(self):
        """Nombres de las features de interacción."""
        return ['budget_delta', 'interest_overlap']

    @property
    def all_context_feature_names(self):
        """Todas las features de contexto (user + match)."""
        return self.user_feature_names + self.match_feature_names
