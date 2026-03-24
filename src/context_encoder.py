"""
SMARTUR Context Encoder v4 (True ML)
Transforma el JSON del formulario React en vectores numéricos para el modelo RF.
Responsable de:
  - Mapeo ordinal de presupuesto y rango de edad
  - Multi-hot encoding de tipos de turismo, y group_type
  - Features de match (budget_delta, interest_overlap, kids_match, romantic_match, tours_match)
"""

# Mapeo de categorías Yelp por tipo de turismo
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
GROUP_TYPES = ['solo', 'pareja', 'familia', 'amigos']

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

class ContextEncoder:
    """Codifica el contexto del turista en features numéricas."""

    def __init__(self):
        self.tourism_types = TOURISM_TYPES
        self.group_types = GROUP_TYPES
        self.budget_map = BUDGET_MAP
        self.age_map = AGE_MAP

    def encode_user(self, context):
        """
        Convierte el dict de contexto del formulario React en features numéricas del usuario.
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

        # One-hot: group_type
        gtype = str(context.get('group_type', 'solo')).lower().strip()
        for g in self.group_types:
            features[f'user_group_{g}'] = 1 if g == gtype else 0

        # Boolean preferences
        features['user_requires_accessibility'] = 1 if context.get('requiere_accesibilidad', False) else 0
        features['user_pref_outdoor'] = 1 if context.get('pref_outdoor', False) else 0
        features['user_wants_tours'] = 1 if context.get('wants_tours', False) else 0
        features['user_needs_hotel'] = 1 if context.get('needs_hotel', False) else 0
        
        # pref_food by default is true (assume they want to see restaurants unless explicitly false)
        features['user_pref_food'] = 1 if context.get('pref_food', True) else 0

        return features

    def _default_user_features(self):
        """Features por defecto cuando no hay contexto (cold start de contexto)."""
        features = {
            'user_budget': 2,
            'user_age_range': 2,
            'user_requires_accessibility': 0,
            'user_pref_outdoor': 0,
            'user_wants_tours': 0,
            'user_needs_hotel': 0,
            'user_pref_food': 1,
        }
        for t in self.tourism_types:
            features[f'user_tur_{t}'] = 0
        for g in self.group_types:
            features[f'user_group_{g}'] = 0
        return features

    def compute_match_features(self, user_features, biz_row):
        """
        Calcula features de interacción user × item de alto nivel para el RandomForest.
        """
        # Budget delta
        user_budget = user_features.get('user_budget', 2)
        price_level = int(biz_row.get('price_level', 2)) if 'price_level' in biz_row else 2
        budget_delta = abs(user_budget - price_level)

        # Interest overlap
        categories_str = str(biz_row.get('categories', '')).lower()
        overlap = 0
        for t in self.tourism_types:
            if user_features.get(f'user_tur_{t}', 0) == 1:
                yelp_cats = MAPEO_CATEGORIAS.get(t, [])
                for cat in yelp_cats:
                    if cat.lower() in categories_str:
                        overlap += 1
                        break

        # Specific contextual matches
        # Kids match: family group + good for kids
        is_good_kids = int(biz_row.get('is_good_for_kids', 0)) if 'is_good_for_kids' in biz_row else 0
        kids_match = 1 if user_features.get('user_group_familia') == 1 and is_good_kids == 1 else 0

        # Romantic match: couple group + romantic/intimate 
        is_rom = int(biz_row.get('is_romantic', 0)) if 'is_romantic' in biz_row else 0
        romantic_match = 1 if user_features.get('user_group_pareja') == 1 and is_rom == 1 else 0

        # Tours match: user wants tours + category includes tours
        tours_match = 1 if user_features.get('user_wants_tours') == 1 and 'tours' in categories_str else 0

        return {
            'budget_delta': budget_delta,
            'interest_overlap': overlap,
            'kids_match': kids_match,
            'romantic_match': romantic_match,
            'tours_match': tours_match,
        }

    def encode_pair(self, context, biz_row):
        """
        Construye el vector completo [User + Item + Interaction] para una pareja usuario-negocio.
        """
        user_feats = self.encode_user(context)
        match_feats = self.compute_match_features(user_feats, biz_row)

        return {**user_feats, **match_feats}

    @property
    def user_feature_names(self):
        """Nombres de las features de usuario (para columnas del DataFrame)."""
        names = ['user_budget', 'user_age_range']
        names += [f'user_tur_{t}' for t in self.tourism_types]
        names += [f'user_group_{g}' for g in self.group_types]
        names += [
            'user_requires_accessibility', 'user_pref_outdoor',
            'user_wants_tours', 'user_needs_hotel', 'user_pref_food'
        ]
        return names

    @property
    def match_feature_names(self):
        """Nombres de las features de interacción."""
        return ['budget_delta', 'interest_overlap', 'kids_match', 'romantic_match', 'tours_match']

    @property
    def all_context_feature_names(self):
        """Todas las features de contexto (user + match)."""
        return self.user_feature_names + self.match_feature_names
