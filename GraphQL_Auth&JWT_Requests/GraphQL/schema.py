import graphene
from graphene_django import DjangoObjectType
from django.contrib.auth.models import User
from .models import Post
import graphql_jwt
from graphql_jwt.decorators import login_required

class PostType(DjangoObjectType):
    class Meta:
        model = Post
        fields = ("id", "title", "body", "author")

class UserType(DjangoObjectType):
    class Meta:
        model = User
        fields = ("id", "username", "email", "posts")

    posts_count = graphene.Int()

    def resolve_posts_count(root, info):
        return root.posts.count()

class CreatePost(graphene.Mutation):
    class Arguments:
        title = graphene.String(required=True)
        body = graphene.String(required=True)

    post = graphene.Field(PostType)

    @login_required
    def mutate(root, info, title, body):
        user = info.context.user
        post = Post(title=title, body=body, author=user)
        post.save()
        return CreatePost(post=post)

class DeletePost(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)

    success = graphene.Boolean()

    @login_required
    def mutate(root, info, id):
        user = info.context.user
        try:
            post = Post.objects.get(pk=id, author=user)
            post.delete()
            return DeletePost(success=True)
        except Post.DoesNotExist:
            raise Exception("Post not found or you are not the author(imposter >:( )")

class CreateUser(graphene.Mutation):
    class Arguments:
        username = graphene.String(required=True)
        email = graphene.String(required=True)
        password = graphene.String(required=True)

    user = graphene.Field(UserType)

    def mutate(root, info, username, email, password):
        user = User(username=username, email=email)
        user.set_password(password)
        user.save()
        return CreateUser(user=user)

class Mutation(graphene.ObjectType):
    token_auth = graphql_jwt.ObtainJSONWebToken.Field()
    verify_token = graphql_jwt.Verify.Field()
    refresh_token = graphql_jwt.Refresh.Field()
    create_user = CreateUser.Field()
    create_post = CreatePost.Field()
    delete_post = DeletePost.Field()

class Query(graphene.ObjectType):
    all_users = graphene.List(UserType)
    all_posts = graphene.List(PostType)

    def resolve_all_users(root, info):
        return User.objects.all()

    def resolve_all_posts(root, info):
        return Post.objects.all()

schema = graphene.Schema(query=Query, mutation=Mutation)
