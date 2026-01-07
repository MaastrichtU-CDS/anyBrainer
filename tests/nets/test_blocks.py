"""Test blocks of layers."""

import pytest
import torch

from anyBrainer.core.networks.blocks import (
    ProjectionHead,
    ClassificationHead,
    FusionHead,
    FPNLightDecoder3D,
    PartialConv,
    MaskedInstanceNorm,
    MultimodalPatchEmbed,
)


@pytest.fixture(scope="module")
def bottleneck_output() -> torch.Tensor:
    """Shape (B, C, D, H, W)"""
    return torch.randn(8, 768, 4, 4, 4)


@pytest.fixture(scope="module")
def bottleneck_output_flat() -> torch.Tensor:
    """Shape (B, C)"""
    return torch.randn(8, 768)


@pytest.fixture(scope="module")
def bottleneck_output_invalid() -> torch.Tensor:
    """Shape (B, C, D, H)"""
    return torch.randn(8, 768, 4, 4)


class TestProjectionHead:
    """Test ProjectionHead in terms of shape, norm, and error handling."""

    @pytest.mark.parametrize("activation", ["ReLU", "GELU", "SiLU"])
    def test_forward(self, bottleneck_output, activation):
        head = ProjectionHead(
            in_dim=768, hidden_dim=2048, proj_dim=128, activation=activation
        )
        output = head(bottleneck_output)
        assert output.shape == (8, 128)
        assert torch.allclose(torch.norm(output, p=2, dim=1), torch.ones(8))

    @pytest.mark.parametrize("activation", ["ReLU", "GELU", "SiLU"])
    def test_norm_output(self, bottleneck_output, activation):
        head = ProjectionHead(
            in_dim=768, hidden_dim=2048, proj_dim=128, activation=activation
        )
        output = head(bottleneck_output)
        assert torch.allclose(torch.norm(output, p=2, dim=1), torch.ones(8))

    def test_forward_flat(self, bottleneck_output_flat):
        head = ProjectionHead(in_dim=768, hidden_dim=2048, proj_dim=128)
        output = head(bottleneck_output_flat)
        assert output.shape == (8, 128)

    def test_forward_error(self, bottleneck_output_invalid):
        with pytest.raises(ValueError):
            head = ProjectionHead(in_dim=768, hidden_dim=2048, proj_dim=128)
            head(bottleneck_output_invalid)

    def test_invalid_activation(self, bottleneck_output):
        with pytest.raises(ValueError):
            head = ProjectionHead(
                in_dim=768, hidden_dim=2048, proj_dim=128, activation="wrong"
            )
            head(bottleneck_output)


class TestClassificationHead:
    """Test ClassificationHead in terms of shape, and error handling."""

    @pytest.mark.parametrize(
        "num_hidden_layers, hidden_dim, activation, activation_kwargs, dropout",
        [
            (0, 0, "ReLU", None, 0.0),
            (1, 1024, "GELU", None, 0.5),
            (2, [512, 64], "SiLU", {"inplace": False}, 0.5),
        ],
    )
    def test_forward(
        self,
        bottleneck_output,
        activation,
        dropout,
        hidden_dim,
        num_hidden_layers,
        activation_kwargs,
    ):
        head = ClassificationHead(
            in_dim=768,
            num_classes=7,
            activation=activation,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            activation_kwargs=activation_kwargs,
        )
        output = head(bottleneck_output)
        assert output.shape == (8, 7)

    def test_forward_flat(self, bottleneck_output_flat):
        head = ClassificationHead(in_dim=768, num_classes=7)
        output = head(bottleneck_output_flat)
        assert output.shape == (8, 7)

    def test_forward_error(self, bottleneck_output_invalid):
        with pytest.raises(ValueError):
            head = ClassificationHead(in_dim=768, num_classes=7)
            head(bottleneck_output_invalid)

    @pytest.mark.parametrize(
        "num_hidden_layers, hidden_dim", [(0, [1024]), (2, [512, 64, 16])]
    )
    def test_error_hidden_dim_args(self, num_hidden_layers, hidden_dim):
        with pytest.raises(ValueError):
            ClassificationHead(
                in_dim=768,
                num_classes=7,
                activation="ReLU",
                dropout=0.5,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
            )

    @pytest.mark.parametrize(
        "num_hidden_layers, dropout", [(1, [0.5]), (2, [0.5, 0.25])]
    )
    def test_error_dropout_args(self, num_hidden_layers, dropout):
        with pytest.raises(ValueError):
            ClassificationHead(
                in_dim=768,
                num_classes=7,
                activation="ReLU",
                dropout=dropout,
                hidden_dim=1024,
                num_hidden_layers=num_hidden_layers,
            )

    @pytest.mark.parametrize(
        "num_hidden_layers, activation", [(1, ["ReLU"]), (2, ["ReLU", "GELU"])]
    )
    def test_error_activation_args(self, num_hidden_layers, activation):
        with pytest.raises(ValueError):
            ClassificationHead(
                in_dim=768,
                num_classes=7,
                activation=activation,
                dropout=0.5,
                hidden_dim=1024,
                num_hidden_layers=num_hidden_layers,
            )

    @pytest.mark.parametrize(
        "activation, activation_kwargs",
        [(["ReLU", "GELU"], [None]), ("SiLU", [{"inplace": False}])],
    )
    def test_error_activation_kwargs_args(self, activation, activation_kwargs):
        with pytest.raises(ValueError):
            ClassificationHead(
                in_dim=768,
                num_classes=7,
                activation=activation,
                dropout=0.5,
                hidden_dim=1024,
                num_hidden_layers=1,
                activation_kwargs=activation_kwargs,
            )


class TestFusionHead:
    """Test FusionHead in terms of shape, and error handling."""

    @pytest.mark.parametrize(
        "input_shape, n_fusion",
        [
            ((4, 4, 8, 768, 4, 4, 4), 4),
            ((4, 8, 768, 4, 4, 4), 1),
            ((4, 768, 4, 4, 4), 1),
        ],
    )
    def test_forward(self, input_shape, n_fusion):
        fusion_head = FusionHead(n_fusion=n_fusion)
        output = fusion_head(torch.randn(input_shape))  # type: ignore
        assert output.shape == (4, 768)

    def test_mod_only(self):
        fusion_head = FusionHead(n_fusion=4, mod_only=True)
        output = fusion_head(torch.randn(4, 4, 768))  # type: ignore
        assert output.shape == (4, 768)

    def test_wrong_shape(self):
        with pytest.raises(ValueError):
            fusion_head = FusionHead(n_fusion=2)
            fusion_head(torch.randn(768, 4, 4, 4))  # type: ignore

    def test_wrong_shape_mod_only(self):
        with pytest.raises(ValueError):
            fusion_head = FusionHead(n_fusion=2, mod_only=True)
            fusion_head(torch.randn(4, 4, 8, 768, 4, 4, 4))  # type: ignore

    def test_mismatch_n_fusion(self):
        with pytest.raises(ValueError):
            fusion_head = FusionHead(n_fusion=2)
            fusion_head(torch.randn(4, 3, 8, 768, 4, 4, 4))  # type: ignore

    def test_mismatch_n_fusion_mod_only(self):
        with pytest.raises(ValueError):
            fusion_head = FusionHead(n_fusion=4, mod_only=True)
            fusion_head(torch.randn(4, 3, 768))  # type: ignore


class TestFPNLightDecoder3D:
    """Test FPNLightDecoder3D in terms of shape, and error handling."""

    @pytest.fixture
    def decoder(self):
        return FPNLightDecoder3D(
            in_feats=[48, 96, 192, 384, 768], in_chans=1, out_channels=2
        )

    @pytest.mark.slow
    def test_forward(self, decoder):
        output = decoder(
            torch.randn(4, 1, 64, 64, 64),
            [
                torch.randn(4, 48, 32, 32, 32),
                torch.randn(4, 96, 16, 16, 16),
                torch.randn(4, 192, 8, 8, 8),
                torch.randn(4, 384, 4, 4, 4),
                torch.randn(4, 768, 2, 2, 2),
            ],
        )
        assert output.shape == (4, 2, 64, 64, 64)

    def test_wrong_shape_feats(self, decoder):
        with pytest.raises(ValueError):
            decoder(
                torch.randn(4, 1, 64, 64, 64),
                [
                    torch.randn(4, 48, 32, 32, 32),
                    torch.randn(4, 96, 16, 16, 16),
                    torch.randn(4, 192, 8, 8, 8),
                ],
            )


class TestPartialConv:
    @pytest.fixture
    def conv(self):
        from monai.networks.blocks.convolutions import Convolution

        return Convolution(
            spatial_dims=3, in_channels=24, out_channels=24, kernel_size=3
        )

    @pytest.fixture
    def input(self):
        return torch.randn(2, 24, 32, 32, 32)

    @pytest.fixture
    def mask(self):
        return torch.randn(2, 24, 32, 32, 32) > 0.5

    def test_matches_shape(self, conv, input, mask):
        out = PartialConv(conv=conv)(input, mask=mask)
        ref_out = conv(input)
        assert out.shape == ref_out.shape

    def test_no_op_mask_none(self, conv, input):
        out = PartialConv(conv=conv)(input)
        ref_out = conv(input)
        assert torch.allclose(out, ref_out)

    def test_norm_values(self, conv, input, mask):
        out = PartialConv(conv=conv)(input, mask=mask)
        ref_out = conv(input)
        assert torch.allclose(out[~mask].mean(), ref_out[~mask].mean(), atol=0.01)
        assert torch.allclose(out[~mask].std(), ref_out[~mask].std(), atol=0.01)


class TestMaskedInstanceNorm:
    @pytest.fixture
    def norm(self):
        return MaskedInstanceNorm(num_features=24)

    @pytest.fixture
    def input(self):
        return torch.randn(2, 24, 32, 32, 32)

    @pytest.fixture
    def mask(self):
        return torch.randn(2, 24, 32, 32, 32) > 0.5

    def test_matches_shape(self, norm, input, mask):
        out = norm(input, mask=mask)
        ref_out = norm(input)
        assert out.shape == ref_out.shape

    def test_no_op_mask_none(self, norm, input):
        out = norm(input)
        ref_out = norm(input)
        assert torch.allclose(out, ref_out)

    def test_norm_values(self, norm, input, mask):
        out = norm(input, mask=mask)
        ref_out = norm(input)
        assert torch.allclose(out[~mask].mean(), ref_out[~mask].mean(), atol=0.01)
        assert torch.allclose(out[~mask].std(), ref_out[~mask].std(), atol=0.01)

    def test_affine(self, norm, input, mask):
        norm = MaskedInstanceNorm(num_features=24, affine=True)
        out = norm(input, mask=mask)
        assert out.shape == input.shape
        assert norm.weight is not None
        assert norm.bias is not None


class TestMultiModalPatchEmbed:
    @pytest.fixture
    def embed(self):
        return MultimodalPatchEmbed(patch_size=2, in_chans=2, embed_dim=24)

    @pytest.fixture
    def input(self):
        return torch.randn(2, 2, 32, 32, 32)

    def test_output_shape(self, embed, input):
        out = embed(input)
        assert out.shape == (2, 24, 16, 16, 16)
        assert len(embed.patch_embeds) == 2

    def test_output_shape_single_channel(self, input):
        embed = MultimodalPatchEmbed(patch_size=2, in_chans=1, embed_dim=24)
        out = embed(input[:, 0:1])
        assert out.shape == (2, 24, 16, 16, 16)
        assert len(embed.patch_embeds) == 1

    def test_fusion(self, input, embed):
        embed_fusion = MultimodalPatchEmbed(
            patch_size=2, in_chans=2, embed_dim=24, fusion="conv1x1"
        )
        out_fusion = embed_fusion(input)
        out_nonfusion = embed(input)
        assert out_fusion.shape == out_nonfusion.shape
        assert not torch.allclose(out_fusion, out_nonfusion, atol=0.01)

    @pytest.mark.parametrize(
        "inject, expected",
        [
            (
                [True, False],
                {"mod1", "mod2"},
            ),  # mod1 and mod2 only for ch0; ignores ch1
            ([True, False], [{"mod1", "mod2"}, {"mod1", "mod2"}]),  # same as above
            (
                [False, True],
                {"mod1", "mod2"},
            ),  # mod1 and mod2 only for ch1; ignores ch0
            ([True, True], {"mod1", "mod2"}),  # mod1 and mod2 for both channels
            (True, [{"mod1", "mod2"}, {"mod1", "mod2"}]),  # same as above
        ],
    )
    def test_modality_tokens_init(self, inject, expected):
        embed = MultimodalPatchEmbed(
            patch_size=2,
            in_chans=2,
            embed_dim=24,
            inject_modality_tokens=inject,
            expected_modalities=expected,
        )
        # Modality tokens for each channels
        if (isinstance(inject, list) and all(inject)) or (
            isinstance(inject, bool) and inject
        ):
            assert len(embed.modality_tokens) == 4
            assert set(embed.modality_tokens.keys()) == {
                "ch0:mod1",
                "ch0:mod2",
                "ch1:mod1",
                "ch1:mod2",
            }
        elif isinstance(inject, list) and inject[0]:
            assert len(embed.modality_tokens) == 2
            assert set(embed.modality_tokens.keys()) == {"ch0:mod1", "ch0:mod2"}
        elif isinstance(inject, list) and inject[1]:
            assert len(embed.modality_tokens) == 2
            assert set(embed.modality_tokens.keys()) == {"ch1:mod1", "ch1:mod2"}

    @pytest.mark.parametrize(
        "inject, expected",
        [
            (
                [True, False],
                {"mod1", "mod2"},
            ),  # mod1 and mod2 only for ch0; ignores ch1
            ([True, False], [{"mod1", "mod2"}, {"mod1", "mod2"}]),  # same as above
            (
                [False, True],
                {"mod1", "mod2"},
            ),  # mod1 and mod2 only for ch1; ignores ch0
            ([True, True], {"mod1", "mod2"}),  # mod1 and mod2 for both channels
            (True, [{"mod1", "mod2"}, {"mod1", "mod2"}]),  # same as above
        ],
    )
    def test_modality_forward(self, inject, expected, embed, input):
        ref_out = embed(input)

        test_embed = MultimodalPatchEmbed(
            patch_size=2,
            in_chans=2,
            embed_dim=24,
            inject_modality_tokens=inject,
            expected_modalities=expected,
        )
        # Modality tokens for each channels
        if (isinstance(inject, list) and all(inject)) or (
            isinstance(inject, bool) and inject
        ):
            out = test_embed(input, modality=["mod1", "mod2"])
            assert out.shape == ref_out.shape
        elif isinstance(inject, list) and inject[0]:
            out = test_embed(input, modality=["mod1", None])
            assert out.shape == ref_out.shape
        elif isinstance(inject, list) and inject[1]:
            out = test_embed(input, modality=[None, "mod2"])
            assert out.shape == ref_out.shape

    @pytest.mark.parametrize(
        "inject, expected",
        [
            ([True], {"mod1", "mod2"}),  # fewer injection locations than in_channels
            ([True, True, True], {"mod1", "mod2"}),  # more injection ...
            (
                [True, True],
                None,
            ),  # no need to inject token if not expecting specific modalities
        ],
    )
    def test_modality_tokens_error_init(self, inject, expected):
        with pytest.raises(ValueError):
            embed = MultimodalPatchEmbed(
                patch_size=2,
                in_chans=2,
                embed_dim=24,
                inject_modality_tokens=inject,
                expected_modalities=expected,
            )

    @pytest.mark.parametrize(
        "modality",
        [
            ["mod1"],  # one modality instead of two
            ["mod2"],  # same
            None,  # modality not provided
        ],
    )
    def test_modality_tokens_ValueError_forward(self, input, modality):
        embed = MultimodalPatchEmbed(
            patch_size=2,
            in_chans=2,
            embed_dim=24,
            inject_modality_tokens=True,
            expected_modalities={"mod1", "mod2"},
        )
        with pytest.raises(ValueError):
            embed(input, modality=modality)

    @pytest.mark.parametrize(
        "modality",
        [
            ["mod3", "mod2"],  # not expecting 'mod3'
        ],
    )
    def test_modality_tokens_RuntimeError_forward(self, input, modality):
        embed = MultimodalPatchEmbed(
            patch_size=2,
            in_chans=2,
            embed_dim=24,
            inject_modality_tokens=True,
            expected_modalities={"mod1", "mod2"},
        )
        with pytest.raises(RuntimeError):
            embed(input, modality=modality)

    @pytest.mark.parametrize(
        "inject, expected",
        [
            ([False, False], None),  # no modality tokens
            (False, {"mod1", "mod2"}),  # same as above
        ],
    )
    def test_modality_tokens_ignore(self, embed, inject, expected, input):
        embed_no_tok = MultimodalPatchEmbed(
            patch_size=2,
            in_chans=2,
            embed_dim=24,
            inject_modality_tokens=inject,
            expected_modalities=expected,
        )
        out_no_tok = embed_no_tok(input)
        out = embed(input)
        assert set(embed_no_tok.modality_tokens.keys()) == set()
        assert out.shape == out_no_tok.shape
